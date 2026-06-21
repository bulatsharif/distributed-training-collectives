"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";

const chunkColors = [
  "#16817a",
  "#c4544f",
  "#bf7c21",
  "#5b8c3a",
  "#7758a6",
  "#2f7fa4",
  "#9a6a2d",
  "#d05b8c",
];

const collectiveDefinitions = {
  allreduce: {
    label: "AllReduce",
    apis: "ncclAllReduce / torch.distributed.all_reduce",
    summary:
      "Every rank contributes a tensor, the operation reduces element-wise across ranks, and every rank receives the same reduced tensor.",
    training:
      "The default DDP gradient path: each replica computes local gradients, all-reduce makes them identical, then every optimizer step stays in sync.",
    metrics: (n) => [
      ["Output", "N elements on every rank"],
      ["Ring bytes per rank", `2 * (N * ${(n - 1)}/${n})`],
      ["Ring phases", `${2 * (n - 1)} neighbor exchanges`],
      ["Sensitive to", "bucket order, rank mapping, stragglers"],
    ],
  },
  reducescatter: {
    label: "ReduceScatter",
    apis: "ncclReduceScatter / torch.distributed.reduce_scatter_tensor",
    summary:
      "Each rank starts with K chunks. Matching chunks are reduced across ranks, then rank i keeps only reduced chunk i.",
    training:
      "Used by FSDP and ZeRO-style sharding to reduce gradients while keeping only each rank's shard.",
    metrics: (n) => [
      ["Output", "N/K elements per rank"],
      ["Ring bytes per rank", `N * ${(n - 1)}/${n}`],
      ["Ring phases", `${n - 1} neighbor exchanges`],
      ["Pairs with", "AllGather to form AllReduce"],
    ],
  },
  allgather: {
    label: "AllGather",
    apis: "ncclAllGather / torch.distributed.all_gather_into_tensor",
    summary:
      "Every rank contributes one chunk. At completion, every rank has the chunks from every rank in rank order.",
    training:
      "FSDP uses all-gather to materialize parameter shards before a layer runs. Tensor parallel layouts use it to assemble partitioned activations.",
    metrics: (n) => [
      ["Output", "K chunks on every rank"],
      ["Ring bytes per rank", `N * ${(n - 1)}/${n}`],
      ["Ring phases", `${n - 1} neighbor exchanges`],
      ["Layout", "rank order matters"],
    ],
  },
  broadcast: {
    label: "Broadcast",
    apis: "ncclBroadcast / torch.distributed.broadcast",
    summary:
      "A root rank owns the source tensor. Every other rank receives an identical copy.",
    training:
      "Used for model state initialization, metadata, random seeds, and root-selected control values.",
    metrics: (n) => [
      ["Output", "root tensor copied to all ranks"],
      ["Tree phases", `${Math.ceil(Math.log2(n))} fan-out levels`],
      ["Root", "logical rank, not device ordinal"],
      ["Sensitive to", "root placement and fan-out path"],
    ],
  },
  reduce: {
    label: "Reduce",
    apis: "ncclReduce / torch.distributed.reduce",
    summary:
      "All ranks contribute tensors. A root rank receives the element-wise reduction; non-root ranks do not receive the reduced result.",
    training:
      "Useful for root-only metrics or summaries when every rank does not need the result.",
    metrics: (n) => [
      ["Output", "N elements on root only"],
      ["Tree phases", `${Math.ceil(Math.log2(n))} fan-in levels`],
      ["Root", "logical rank, not device ordinal"],
      ["Equivalent", "first half of Broadcast + Reduce"],
    ],
  },
  alltoall: {
    label: "AllToAll",
    apis: "ncclAlltoAll / torch.distributed.all_to_all",
    summary:
      "Every rank splits its input into one chunk per destination. Rank j receives the j-labeled chunk from every source.",
    training:
      "Core primitive for MoE token exchange, sequence-parallel transposes, and distributed shuffles.",
    metrics: (n) => [
      ["Output", "one chunk from each source"],
      ["Peer exchanges", `${n * (n - 1)} directed sends`],
      ["Payload shape", "K destination chunks per rank"],
      ["Sensitive to", "load balance and network bisection"],
    ],
  },
  gather: {
    label: "Gather",
    apis: "ncclGather / torch.distributed.gather",
    summary:
      "Every rank contributes one chunk. The root rank receives all chunks in rank order.",
    training:
      "Useful for root-side validation summaries, predictions, or small control data.",
    metrics: (n) => [
      ["Output", "K chunks on root"],
      ["Tree phases", `${Math.ceil(Math.log2(n))} fan-in levels`],
      ["Layout", "rank order matters"],
      ["Root pressure", "root stores K*N elements"],
    ],
  },
  scatter: {
    label: "Scatter",
    apis: "ncclScatter / torch.distributed.scatter",
    summary:
      "A root rank starts with K chunks and sends chunk i to rank i.",
    training:
      "Used for root-driven partitioning, setup data, and simple custom distribution paths.",
    metrics: (n) => [
      ["Output", "one chunk per rank"],
      ["Tree phases", `${Math.ceil(Math.log2(n))} fan-out levels`],
      ["Layout", "rank order matters"],
      ["Root pressure", "root starts with K*N elements"],
    ],
  },
  barrier: {
    label: "Barrier",
    apis: "torch.distributed.barrier / NCCL signal-wait patterns",
    summary:
      "All ranks wait until every participant reaches the same phase. There is no meaningful tensor payload.",
    training:
      "Used sparingly for phase boundaries, debugging, and making failures easier to localize.",
    metrics: (n) => [
      ["Output", "no tensor payload"],
      ["Participants", `${n} ranks must arrive`],
      ["Cost", "latency plus straggler wait"],
      ["Risk", "can hide the real earlier error"],
    ],
  },
};

const algorithmOptions = [
  { value: "ring", label: "Ring / chunked" },
  { value: "tree", label: "Tree / fan-in-out" },
  { value: "direct", label: "Direct peer view" },
];

const state = {
  collective: "allreduce",
  algorithm: "ring",
  ranks: 6,
  step: 0,
  timer: null,
};

const els = {
  collectiveSelect: document.querySelector("#collective-select"),
  algorithmSelect: document.querySelector("#algorithm-select"),
  rankSlider: document.querySelector("#rank-slider"),
  rankOutput: document.querySelector("#rank-output"),
  prevStep: document.querySelector("#prev-step"),
  nextStep: document.querySelector("#next-step"),
  playSteps: document.querySelector("#play-steps"),
  svg: document.querySelector("#collective-svg"),
  stepTitle: document.querySelector("#step-title"),
  stepDetail: document.querySelector("#step-detail"),
  stepCounter: document.querySelector("#step-counter"),
  timeline: document.querySelector("#timeline"),
  metricGrid: document.querySelector("#metric-grid"),
  collectiveSummary: document.querySelector("#collective-summary"),
  trainingUse: document.querySelector("#training-use"),
  heroNetwork: document.querySelector("#hero-network"),
  topologySvg: document.querySelector("#topology-svg"),
};

function createSvgElement(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  if (text !== null) {
    node.textContent = text;
  }
  return node;
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function polarPoint(cx, cy, radius, index, total, offset = -Math.PI / 2) {
  const angle = offset + (Math.PI * 2 * index) / total;
  return {
    x: cx + Math.cos(angle) * radius,
    y: cy + Math.sin(angle) * radius,
    angle,
  };
}

function rankPositions(n, cx = 450, cy = 300, radius = 205) {
  return Array.from({ length: n }, (_, i) => polarPoint(cx, cy, radius, i, n));
}

function ringEdges(n, phase = "neutral") {
  return Array.from({ length: n }, (_, i) => ({
    from: i,
    to: (i + 1) % n,
    label: `r${i} → r${(i + 1) % n}`,
    phase,
  }));
}

function reverseRingEdges(n, phase = "neutral") {
  return Array.from({ length: n }, (_, i) => ({
    from: i,
    to: (i - 1 + n) % n,
    label: `r${i} → r${(i - 1 + n) % n}`,
    phase,
  }));
}

function treeLevelEdges(n, level, direction, root = 0) {
  const edges = [];
  const span = 2 ** level;
  for (let i = 0; i < n; i += span * 2) {
    const parent = (root + i) % n;
    const child = (root + i + span) % n;
    if (i + span < n) {
      edges.push(
        direction === "in"
          ? { from: child, to: parent, label: `r${child} → r${parent}`, phase: "reduce" }
          : { from: parent, to: child, label: `r${parent} → r${child}`, phase: "gather" },
      );
    }
  }
  return edges;
}

function fullPeerEdges(n) {
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j) edges.push({ from: i, to: j, label: `r${i} → r${j}`, phase: "gather" });
    }
  }
  return edges;
}

function rootEdges(n, direction, root = 0) {
  return Array.from({ length: n }, (_, i) => i)
    .filter((i) => i !== root)
    .map((i) =>
      direction === "out"
        ? { from: root, to: i, label: `root → r${i}`, phase: "gather" }
        : { from: i, to: root, label: `r${i} → root`, phase: "reduce" },
    );
}

function makeChunks(n, mode, stepIndex = 0) {
  return Array.from({ length: n }, (_, rank) => {
    if (mode === "all") return Array.from({ length: n }, (_, i) => i);
    if (mode === "own") return [rank];
    if (mode === "root") return rank === 0 ? Array.from({ length: n }, (_, i) => i) : [];
    if (mode === "reducedShard") return [rank];
    if (mode === "growing") {
      const count = Math.min(n, stepIndex + 1);
      return Array.from({ length: count }, (_, j) => (rank - j + n) % n).sort((a, b) => a - b);
    }
    if (mode === "broadcastProgress") {
      return rank === 0 || rank <= stepIndex ? Array.from({ length: n }, (_, i) => i) : [];
    }
    if (mode === "scatterProgress") return rank === 0 || rank <= stepIndex ? [rank] : [];
    return [];
  });
}

function buildSteps(collective, algorithm, n) {
  const logN = Math.ceil(Math.log2(n));
  const def = collectiveDefinitions[collective];

  if (collective === "barrier") {
    return [
      {
        title: "Ranks enter the barrier",
        detail:
          "Each rank records that it has reached the phase boundary. Fast ranks wait for the slowest participant.",
        edges: fullPeerEdges(n).slice(0, Math.min(n * 2, 14)),
        chunks: makeChunks(n, "own"),
      },
      {
        title: "All arrivals observed",
        detail:
          "The barrier releases once every rank has arrived. This is useful for debugging but should not be used to paper over ordering bugs.",
        edges: [],
        chunks: makeChunks(n, "own"),
      },
    ];
  }

  if (algorithm === "tree" || ["broadcast", "reduce", "gather", "scatter"].includes(collective)) {
    if (collective === "broadcast" || collective === "scatter") {
      return Array.from({ length: logN + 1 }, (_, i) => ({
        title: i === 0 ? `${def.label}: root owns the payload` : `Fan-out level ${i}`,
        detail:
          i === 0
            ? "The root rank starts with the source buffer. The root is a logical rank in the communicator."
            : "Ranks that already have data forward it to ranks that have not received it yet.",
        edges: i === 0 ? [] : treeLevelEdges(n, i - 1, "out"),
        chunks:
          collective === "broadcast"
            ? makeChunks(n, i === 0 ? "root" : "broadcastProgress", i)
            : makeChunks(n, "scatterProgress", i),
      }));
    }

    if (collective === "reduce" || collective === "gather") {
      const steps = Array.from({ length: logN }, (_, i) => ({
        title: `Fan-in level ${i + 1}`,
        detail:
          collective === "reduce"
            ? "Pairs of ranks combine matching tensor elements and forward the partial reduction toward the root."
            : "Ranks forward their chunks toward the root. The root's receive buffer is ordered by rank.",
        edges: treeLevelEdges(n, i, "in"),
        chunks: makeChunks(n, "own"),
      }));
      steps.push({
        title: `${def.label}: root has the result`,
        detail:
          collective === "reduce"
            ? "Only the root rank owns the reduced tensor. Other ranks can continue without receiving a copy."
            : "Only the root rank owns the gathered K-chunk output.",
        edges: [],
        chunks: makeChunks(n, "root"),
      });
      return steps;
    }

    if (collective === "allreduce") {
      const reduce = Array.from({ length: logN }, (_, i) => ({
        title: `Tree reduce level ${i + 1}`,
        detail: "Partial reductions move up the tree toward the root.",
        edges: treeLevelEdges(n, i, "in"),
        chunks: makeChunks(n, "own"),
      }));
      const bcast = Array.from({ length: logN }, (_, i) => ({
        title: `Tree broadcast level ${i + 1}`,
        detail: "The reduced result fans back out from the root to every rank.",
        edges: treeLevelEdges(n, i, "out"),
        chunks: makeChunks(n, i + 1 >= logN ? "all" : "broadcastProgress", i + 1),
      }));
      return [
        ...reduce,
        {
          title: "Root owns the reduction",
          detail: "The reduce phase has produced the complete reduced tensor at the root.",
          edges: [],
          chunks: makeChunks(n, "root"),
        },
        ...bcast,
      ];
    }
  }

  if (algorithm === "direct" || collective === "alltoall") {
    if (collective === "alltoall") {
      return [
        {
          title: "Each rank partitions by destination",
          detail:
            "Rank i has K chunks: one destined for every rank. Chunk labels in the diagram represent destination owners.",
          edges: [],
          chunks: makeChunks(n, "all"),
        },
        {
          title: "Every source exchanges with every destination",
          detail:
            "All-to-all is dense peer exchange. In practice implementations schedule these transfers to avoid oversubscribing links.",
          edges: fullPeerEdges(n),
          chunks: makeChunks(n, "all"),
        },
        {
          title: "Each rank owns chunks addressed to it",
          detail:
            "Rank j receives the j-labeled chunk from every source. Load imbalance in chunk sizes creates stragglers.",
          edges: [],
          chunks: makeChunks(n, "all"),
        },
      ];
    }

    if (collective === "allgather") {
      return [
        {
          title: "Each rank starts with one shard",
          detail: "Every rank contributes its local shard.",
          edges: [],
          chunks: makeChunks(n, "own"),
        },
        {
          title: "Peers share shards",
          detail: "All ranks exchange shards until every rank has a full rank-ordered buffer.",
          edges: fullPeerEdges(n).slice(0, Math.min(n * 3, 24)),
          chunks: makeChunks(n, "growing", Math.floor(n / 2)),
        },
        {
          title: "All ranks own the full gathered tensor",
          detail: "The output is identical on every rank and ordered by source rank.",
          edges: [],
          chunks: makeChunks(n, "all"),
        },
      ];
    }
  }

  if (collective === "reducescatter") {
    return [
      {
        title: "Each rank starts with K chunks",
        detail:
          "The input buffer is logically split into one reduce destination per rank. Matching chunk indexes will be reduced together.",
        edges: [],
        chunks: makeChunks(n, "all"),
      },
      ...Array.from({ length: n - 1 }, (_, i) => ({
        title: `Reduce-scatter hop ${i + 1}`,
        detail:
          "Each rank sends a chunk to its next neighbor and receives one from its previous neighbor, accumulating partial reductions for the shard it will own.",
        edges: ringEdges(n, "reduce"),
        chunks: makeChunks(n, "growing", i),
      })),
      {
        title: "Each rank owns one reduced shard",
        detail: "The reduced tensor is partitioned: rank i owns the i-th reduced chunk.",
        edges: [],
        chunks: makeChunks(n, "reducedShard"),
      },
    ];
  }

  if (collective === "allgather") {
    return [
      {
        title: "Each rank starts with one shard",
        detail: "The all-gather begins from sharded ownership.",
        edges: [],
        chunks: makeChunks(n, "own"),
      },
      ...Array.from({ length: n - 1 }, (_, i) => ({
        title: `All-gather hop ${i + 1}`,
        detail:
          "Ranks forward the shards they have received so far. After enough hops, every shard has visited every rank.",
        edges: ringEdges(n, "gather"),
        chunks: makeChunks(n, "growing", i + 1),
      })),
      {
        title: "Every rank has every shard",
        detail: "The gathered output is K shards in rank order on every rank.",
        edges: [],
        chunks: makeChunks(n, "all"),
      },
    ];
  }

  if (collective === "allreduce") {
    return [
      {
        title: "Each rank starts with local gradients",
        detail:
          "The tensor is split into chunks. Every rank owns a full local copy before communication starts.",
        edges: [],
        chunks: makeChunks(n, "all"),
      },
      ...Array.from({ length: n - 1 }, (_, i) => ({
        title: `Reduce-scatter hop ${i + 1}`,
        detail:
          "Neighbor exchanges circulate chunks while ranks accumulate partial sums. At the end of this phase, each rank owns one reduced shard.",
        edges: ringEdges(n, "reduce"),
        chunks: makeChunks(n, "growing", i),
      })),
      {
        title: "Reduced shards are distributed",
        detail:
          "All reduction work is complete, but no rank has the full tensor yet. This is the point where sharded optimizers can stop.",
        edges: [],
        chunks: makeChunks(n, "reducedShard"),
      },
      ...Array.from({ length: n - 1 }, (_, i) => ({
        title: `All-gather hop ${i + 1}`,
        detail:
          "Reduced shards circulate around the ring. Each hop gives every rank another reduced shard.",
        edges: ringEdges(n, "gather"),
        chunks: makeChunks(n, "growing", i + 1),
      })),
      {
        title: "All ranks have identical reduced tensors",
        detail:
          "The all-reduce is complete. Every rank can apply the same optimizer update.",
        edges: [],
        chunks: makeChunks(n, "all"),
      },
    ];
  }

  if (collective === "broadcast") {
    return [
      {
        title: "Root owns the source buffer",
        detail: "Rank 0 is the source in this visualization.",
        edges: [],
        chunks: makeChunks(n, "root"),
      },
      {
        title: "Root sends to peers",
        detail:
          "A direct view shows all peers receiving from root. Real implementations may use tree or pipeline schedules.",
        edges: rootEdges(n, "out"),
        chunks: makeChunks(n, "broadcastProgress", n - 1),
      },
      {
        title: "Every rank has the root buffer",
        detail: "All ranks now own an identical copy.",
        edges: [],
        chunks: makeChunks(n, "all"),
      },
    ];
  }

  if (collective === "reduce") {
    return [
      {
        title: "Every rank contributes",
        detail: "All ranks start with local tensors.",
        edges: [],
        chunks: makeChunks(n, "own"),
      },
      {
        title: "Peers reduce into root",
        detail: "A direct view shows all ranks sending to root. Real implementations usually schedule a tree.",
        edges: rootEdges(n, "in"),
        chunks: makeChunks(n, "own"),
      },
      {
        title: "Root owns the reduced tensor",
        detail: "Only the root rank receives the final reduction.",
        edges: [],
        chunks: makeChunks(n, "root"),
      },
    ];
  }

  if (collective === "gather") {
    return [
      {
        title: "Every rank owns one shard",
        detail: "Gather is a many-to-one data movement.",
        edges: [],
        chunks: makeChunks(n, "own"),
      },
      {
        title: "Peers send shards to root",
        detail: "The root's output buffer is ordered by source rank.",
        edges: rootEdges(n, "in"),
        chunks: makeChunks(n, "own"),
      },
      {
        title: "Root has all shards",
        detail: "Only the root rank owns the gathered result.",
        edges: [],
        chunks: makeChunks(n, "root"),
      },
    ];
  }

  if (collective === "scatter") {
    return [
      {
        title: "Root owns all shards",
        detail: "Scatter is a one-to-many data movement.",
        edges: [],
        chunks: makeChunks(n, "root"),
      },
      {
        title: "Root sends each shard to its owner",
        detail: "Rank i receives chunk i.",
        edges: rootEdges(n, "out"),
        chunks: makeChunks(n, "scatterProgress", n - 1),
      },
      {
        title: "Each rank owns one shard",
        detail: "The root's large buffer has been partitioned across ranks.",
        edges: [],
        chunks: makeChunks(n, "own"),
      },
    ];
  }

  return [
    {
      title: def.label,
      detail: def.summary,
      edges: ringEdges(n),
      chunks: makeChunks(n, "own"),
    },
  ];
}

function renderEdge(svg, positions, edge, index, totalEdges) {
  const from = positions[edge.from];
  const to = positions[edge.to];
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const offset = (index % 3) * 4 - 4;
  const ox = -uy * offset;
  const oy = ux * offset;
  const startX = from.x + ux * 45 + ox;
  const startY = from.y + uy * 45 + oy;
  const endX = to.x - ux * 45 + ox;
  const endY = to.y - uy * 45 + oy;
  const curve = totalEdges > 12 ? 16 : 24;
  const mx = (startX + endX) / 2 - uy * curve;
  const my = (startY + endY) / 2 + ux * curve;
  const path = `M ${startX} ${startY} Q ${mx} ${my} ${endX} ${endY}`;
  svg.appendChild(
    createSvgElement("path", {
      d: path,
      class: `edge active ${edge.phase || ""}`,
      markerEnd: "url(#arrowhead)",
    }),
  );
  const packet = createSvgElement("circle", {
    cx: (startX + endX) / 2,
    cy: (startY + endY) / 2,
    r: 8,
    fill: chunkColors[(edge.from + edge.to) % chunkColors.length],
    class: "packet",
  });
  svg.appendChild(packet);
}

function renderRank(svg, position, rank, chunks, totalRanks) {
  const g = createSvgElement("g", { class: "rank-node" });
  g.appendChild(createSvgElement("circle", { cx: position.x, cy: position.y, r: 37 }));
  g.appendChild(createSvgElement("text", { x: position.x, y: position.y - 5 }, `r${rank}`));
  g.appendChild(
    createSvgElement("text", { x: position.x, y: position.y + 18, class: "svg-small" }, `GPU ${rank}`),
  );

  const max = Math.max(totalRanks, 1);
  const barWidth = 68;
  const chunkWidth = Math.max(6, (barWidth - (max - 1) * 3) / max);
  const y = position.y + 50;
  const x = position.x - barWidth / 2;
  chunks.forEach((chunk) => {
    g.appendChild(
      createSvgElement("rect", {
        x: x + chunk * (chunkWidth + 3),
        y,
        width: chunkWidth,
        height: 12,
        rx: 3,
        fill: chunkColors[chunk % chunkColors.length],
        class: "chunk",
      }),
    );
  });
  g.appendChild(
    createSvgElement("rect", {
      x,
      y,
      width: barWidth,
      height: 12,
      rx: 3,
      fill: "none",
      stroke: "#b8c4c1",
      "stroke-width": 1,
    }),
  );
  svg.appendChild(g);
}

function renderLegend(svg, n) {
  const g = createSvgElement("g", { transform: "translate(28 552)" });
  g.appendChild(createSvgElement("text", { x: 0, y: 0, class: "svg-label" }, "Chunk labels"));
  for (let i = 0; i < n; i++) {
    const x = i * 74;
    g.appendChild(
      createSvgElement("rect", {
        x,
        y: 18,
        width: 20,
        height: 14,
        rx: 3,
        fill: chunkColors[i % chunkColors.length],
        class: "chunk",
      }),
    );
    g.appendChild(createSvgElement("text", { x: x + 26, y: 30, class: "svg-small" }, `chunk ${i}`));
  }
  svg.appendChild(g);
}

function renderSimulator() {
  const def = collectiveDefinitions[state.collective];
  const steps = buildSteps(state.collective, state.algorithm, state.ranks);
  state.step = Math.max(0, Math.min(state.step, steps.length - 1));
  const step = steps[state.step];
  const positions = rankPositions(state.ranks);

  clear(els.svg);
  const defs = createSvgElement("defs");
  const marker = createSvgElement("marker", {
    id: "arrowhead",
    viewBox: "0 0 10 10",
    refX: "9",
    refY: "5",
    markerWidth: "7",
    markerHeight: "7",
    orient: "auto-start-reverse",
  });
  marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "currentColor" }));
  defs.appendChild(marker);
  els.svg.appendChild(defs);

  const baseRing = ringEdges(state.ranks);
  baseRing.forEach((edge) => {
    const from = positions[edge.from];
    const to = positions[edge.to];
    els.svg.appendChild(
      createSvgElement("line", {
        x1: from.x,
        y1: from.y,
        x2: to.x,
        y2: to.y,
        class: "edge",
      }),
    );
  });

  step.edges.forEach((edge, index) => renderEdge(els.svg, positions, edge, index, step.edges.length));
  positions.forEach((position, rank) => renderRank(els.svg, position, rank, step.chunks[rank] || [], state.ranks));
  renderLegend(els.svg, state.ranks);

  els.stepTitle.textContent = step.title;
  els.stepDetail.textContent = step.detail;
  els.stepCounter.textContent = `${state.step + 1} / ${steps.length}`;
  els.collectiveSummary.textContent = `${def.summary} API shape: ${def.apis}.`;
  els.trainingUse.textContent = def.training;
  renderMetrics(def.metrics(state.ranks));
  renderTimeline(steps.length);
}

function renderMetrics(metrics) {
  clear(els.metricGrid);
  metrics.forEach(([label, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    metric.append(span, strong);
    els.metricGrid.appendChild(metric);
  });
}

function renderTimeline(count) {
  clear(els.timeline);
  for (let i = 0; i < count; i++) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = i === state.step ? "active" : "";
    button.setAttribute("aria-label", `Go to step ${i + 1}`);
    button.addEventListener("click", () => {
      state.step = i;
      stopPlayback();
      renderSimulator();
    });
    els.timeline.appendChild(button);
  }
}

function stopPlayback() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  els.playSteps.textContent = "Play";
}

function nextStep() {
  const steps = buildSteps(state.collective, state.algorithm, state.ranks);
  state.step = (state.step + 1) % steps.length;
  renderSimulator();
}

function previousStep() {
  const steps = buildSteps(state.collective, state.algorithm, state.ranks);
  state.step = (state.step - 1 + steps.length) % steps.length;
  renderSimulator();
}

function updateAlgorithmAvailability() {
  const treePreferred = ["broadcast", "reduce", "gather", "scatter", "barrier"];
  Array.from(els.algorithmSelect.options).forEach((option) => {
    option.disabled = false;
  });
  if (treePreferred.includes(state.collective) && state.algorithm === "ring") {
    state.algorithm = "tree";
    els.algorithmSelect.value = state.algorithm;
  }
}

function initControls() {
  Object.entries(collectiveDefinitions).forEach(([value, def]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = def.label;
    els.collectiveSelect.appendChild(option);
  });
  algorithmOptions.forEach((algorithm) => {
    const option = document.createElement("option");
    option.value = algorithm.value;
    option.textContent = algorithm.label;
    els.algorithmSelect.appendChild(option);
  });
  els.collectiveSelect.value = state.collective;
  els.algorithmSelect.value = state.algorithm;

  els.collectiveSelect.addEventListener("change", () => {
    state.collective = els.collectiveSelect.value;
    state.step = 0;
    updateAlgorithmAvailability();
    stopPlayback();
    renderSimulator();
  });
  els.algorithmSelect.addEventListener("change", () => {
    state.algorithm = els.algorithmSelect.value;
    state.step = 0;
    stopPlayback();
    renderSimulator();
  });
  els.rankSlider.addEventListener("input", () => {
    state.ranks = Number(els.rankSlider.value);
    els.rankOutput.textContent = String(state.ranks);
    state.step = 0;
    stopPlayback();
    renderSimulator();
  });
  els.prevStep.addEventListener("click", () => {
    stopPlayback();
    previousStep();
  });
  els.nextStep.addEventListener("click", () => {
    stopPlayback();
    nextStep();
  });
  els.playSteps.addEventListener("click", () => {
    if (state.timer) {
      stopPlayback();
      return;
    }
    els.playSteps.textContent = "Pause";
    state.timer = setInterval(nextStep, 1150);
  });
}

function renderHeroNetwork() {
  const svg = els.heroNetwork;
  clear(svg);
  const defs = createSvgElement("defs");
  const marker = createSvgElement("marker", {
    id: "hero-arrow",
    viewBox: "0 0 10 10",
    refX: "9",
    refY: "5",
    markerWidth: "6",
    markerHeight: "6",
    orient: "auto",
  });
  marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#16817a" }));
  defs.appendChild(marker);
  svg.appendChild(defs);

  const positions = rankPositions(8, 380, 225, 145);
  reverseRingEdges(8, "gather").forEach((edge, index) => {
    const from = positions[edge.from];
    const to = positions[edge.to];
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const len = Math.hypot(dx, dy);
    const ux = dx / len;
    const uy = dy / len;
    const sx = from.x + ux * 34;
    const sy = from.y + uy * 34;
    const ex = to.x - ux * 34;
    const ey = to.y - uy * 34;
    svg.appendChild(
      createSvgElement("line", {
        x1: sx,
        y1: sy,
        x2: ex,
        y2: ey,
        stroke: index % 2 ? "#16817a" : "#5b8c3a",
        "stroke-width": 4,
        "stroke-linecap": "round",
        "marker-end": "url(#hero-arrow)",
        opacity: 0.86,
      }),
    );
  });

  svg.appendChild(
    createSvgElement("text", { x: 380, y: 54, class: "svg-label", "text-anchor": "middle" }, "ring all-reduce"),
  );
  svg.appendChild(
    createSvgElement("text", { x: 380, y: 86, class: "svg-small", "text-anchor": "middle" }, "reduce-scatter + all-gather"),
  );
  positions.forEach((position, rank) => {
    const g = createSvgElement("g");
    g.appendChild(createSvgElement("circle", { cx: position.x, cy: position.y, r: 30, fill: "#fff", stroke: "#20252a", "stroke-width": 2 }));
    g.appendChild(createSvgElement("text", { x: position.x, y: position.y + 5, "text-anchor": "middle", "font-weight": 800, fill: "#20252a" }, `r${rank}`));
    for (let i = 0; i < 3; i++) {
      g.appendChild(
        createSvgElement("rect", {
          x: position.x - 23 + i * 16,
          y: position.y + 39,
          width: 12,
          height: 10,
          rx: 2,
          fill: chunkColors[(rank + i) % chunkColors.length],
        }),
      );
    }
    svg.appendChild(g);
  });

  const box = createSvgElement("g", { transform: "translate(252 360)" });
  box.appendChild(createSvgElement("rect", { x: 0, y: 0, width: 256, height: 62, rx: 8, fill: "#eef3ef", stroke: "#d7dfdc" }));
  box.appendChild(createSvgElement("text", { x: 128, y: 25, "text-anchor": "middle", class: "svg-label" }, "same gradient result"));
  box.appendChild(createSvgElement("text", { x: 128, y: 45, "text-anchor": "middle", class: "svg-small" }, "every rank can step the optimizer"));
  svg.appendChild(box);
}

function renderTopology() {
  const svg = els.topologySvg;
  clear(svg);
  const nodeXs = [185, 715];
  nodeXs.forEach((x, node) => {
    svg.appendChild(createSvgElement("rect", { x: x - 155, y: 65, width: 310, height: 320, rx: 8, fill: "#ffffff", stroke: "#cbd7d4" }));
    svg.appendChild(createSvgElement("text", { x, y: 96, "text-anchor": "middle", class: "svg-label" }, `node ${node}`));
    for (let i = 0; i < 4; i++) {
      const gx = x - 105 + (i % 2) * 210;
      const gy = 145 + Math.floor(i / 2) * 112;
      svg.appendChild(createSvgElement("rect", { x: gx - 44, y: gy - 28, width: 88, height: 56, rx: 8, fill: "#f8faf8", stroke: "#20252a", "stroke-width": 2 }));
      svg.appendChild(createSvgElement("text", { x: gx, y: gy + 5, "text-anchor": "middle", "font-weight": 800, fill: "#20252a" }, `GPU ${i}`));
    }
    svg.appendChild(createSvgElement("line", { x1: x - 105, y1: 145, x2: x + 105, y2: 145, stroke: "#16817a", "stroke-width": 5, "stroke-linecap": "round" }));
    svg.appendChild(createSvgElement("line", { x1: x - 105, y1: 257, x2: x + 105, y2: 257, stroke: "#16817a", "stroke-width": 5, "stroke-linecap": "round" }));
    svg.appendChild(createSvgElement("line", { x1: x - 105, y1: 145, x2: x - 105, y2: 257, stroke: "#16817a", "stroke-width": 5, "stroke-linecap": "round" }));
    svg.appendChild(createSvgElement("line", { x1: x + 105, y1: 145, x2: x + 105, y2: 257, stroke: "#16817a", "stroke-width": 5, "stroke-linecap": "round" }));
    svg.appendChild(createSvgElement("rect", { x: x - 70, y: 314, width: 140, height: 42, rx: 8, fill: "#273033" }));
    svg.appendChild(createSvgElement("text", { x, y: 341, "text-anchor": "middle", fill: "#fff", "font-weight": 760 }, "NIC / HCA"));
  });
  svg.appendChild(createSvgElement("line", { x1: 255, y1: 335, x2: 645, y2: 335, stroke: "#bf7c21", "stroke-width": 7, "stroke-dasharray": "12 10", "stroke-linecap": "round" }));
  svg.appendChild(createSvgElement("text", { x: 450, y: 318, "text-anchor": "middle", class: "svg-label" }, "InfiniBand / Ethernet fabric"));
  svg.appendChild(createSvgElement("text", { x: 450, y: 428, "text-anchor": "middle", class: "svg-small" }, "NCCL maps logical ranks onto these physical paths, then slices collectives into rings, trees, and channels."));
}

initControls();
renderHeroNetwork();
renderTopology();
renderSimulator();
