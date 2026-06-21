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

const collectiveWatch = {
  allreduce:
    "For large tensors, ring all-reduce is bandwidth-oriented but exposes every slow rank. Bucket ordering decides whether communication is hidden behind remaining backward compute or appears on the critical path.",
  reducescatter:
    "The final result is intentionally incomplete on each rank. That is a feature for sharded optimizers, but a bug if later code expects the full gradient tensor locally.",
  allgather:
    "All-gather increases resident memory because every rank materializes every shard. FSDP systems schedule and free gathered parameters carefully to avoid spikes.",
  broadcast:
    "The root is a logical communicator rank. If ranks disagree on the root or call order, the operation can hang even when tensor shapes look valid.",
  reduce:
    "Only the root receives the reduced result. Use all-reduce when every rank needs the value for control flow, logging, or an optimizer decision.",
  alltoall:
    "The average byte count is not enough. One overloaded destination expert or one much larger token block can make all other ranks wait.",
  gather:
    "Gather concentrates memory and receive work on the root. It is fine for small evaluation outputs but can become a root bottleneck for large tensors.",
  scatter:
    "Scatter assumes the root already owns correctly partitioned chunks. A shape or ordering mismatch silently sends the wrong shard to the wrong rank.",
  barrier:
    "A barrier tells you that ranks met at the same point; it does not fix the earlier mismatch that caused them to diverge.",
};

const algorithmOptions = [
  { value: "ring", label: "Ring / chunked" },
  { value: "tree", label: "Tree / fan-in-out" },
  { value: "direct", label: "Direct peer view" },
];

const parallelismDefinitions = {
  data: {
    label: "Data Parallel / DDP",
    summary:
      "The full model is replicated on every rank, while each rank consumes a different data shard. Training semantics stay simple: every replica should apply the same parameter update.",
    communication:
      "Backward produces local gradients, then the data-parallel group synchronizes them with all-reduce or an equivalent reduce-scatter plus all-gather path.",
    fit:
      "Best when the model fits per GPU and the main goal is throughput from larger global batch size.",
    metrics: [
      ["Model state", "replicated"],
      ["Batch", "sharded by sample"],
      ["Main traffic", "gradient all-reduce"],
      ["Scaling limit", "batch size and gradient sync"],
    ],
  },
  fsdp: {
    label: "FSDP / ZeRO-style Sharded DP",
    summary:
      "The data-parallel axis remains, but model state is sharded across data-parallel ranks instead of fully replicated. Each rank materializes parameters only when needed.",
    communication:
      "Forward and backward all-gather parameter shards before compute; backward reduce-scatters gradients back into owned shards.",
    fit:
      "Best when the model is close to fitting but optimizer state, gradients, or parameters are the memory bottleneck.",
    metrics: [
      ["Model state", "parameters, gradients, optimizer sharded"],
      ["Batch", "sharded by sample"],
      ["Main traffic", "all-gather + reduce-scatter"],
      ["Scaling limit", "parameter gather exposure"],
    ],
  },
  tensor: {
    label: "Tensor Parallel",
    summary:
      "A single layer's matrices and intermediate tensors are split across ranks. Ranks jointly compute one layer rather than owning independent replicas of the whole layer.",
    communication:
      "Column/row splits require all-reduce, all-gather, or reduce-scatter around linear layers, attention projections, and partial sums.",
    fit:
      "Best for layers that are too large or too slow on one GPU, especially within a high-bandwidth NVLink or NVSwitch domain.",
    metrics: [
      ["Model state", "layer tensors sharded"],
      ["Batch", "usually shared inside TP group"],
      ["Main traffic", "partial-sum collectives"],
      ["Scaling limit", "intra-layer latency"],
    ],
  },
  pipeline: {
    label: "Pipeline Parallel",
    summary:
      "The model depth is split into consecutive stages. Microbatches flow through the stages so different GPUs work on different layers at the same time.",
    communication:
      "Adjacent stages send activations forward and activation gradients backward with point-to-point communication.",
    fit:
      "Best when total model depth does not fit on one rank, or when cross-node bandwidth makes full-graph collectives too expensive.",
    metrics: [
      ["Model state", "layers sharded by depth"],
      ["Batch", "split into microbatches"],
      ["Main traffic", "activation send/recv"],
      ["Scaling limit", "pipeline bubble and imbalance"],
    ],
  },
  expert: {
    label: "Expert Parallel",
    summary:
      "MoE expert weights are split across ranks. Dense layers may remain replicated or separately parallelized, while only router-selected tokens visit each expert.",
    communication:
      "MoE layers dispatch tokens to expert owners and combine outputs, commonly with all-to-all traffic inside an expert-parallel group.",
    fit:
      "Best for sparse MoE models where total parameter count grows through many experts but each token activates only a subset.",
    metrics: [
      ["Model state", "experts sharded"],
      ["Token path", "router -> expert owner"],
      ["Main traffic", "all-to-all dispatch/combine"],
      ["Scaling limit", "token balance and bisection"],
    ],
  },
  expertData: {
    label: "Expert Data Parallel",
    summary:
      "Expert-parallel groups are replicated across data-parallel replicas. Each replica routes its own tokens to local expert owners, while matching experts synchronize across replicas.",
    communication:
      "All-to-all token dispatch happens inside each expert-parallel group; expert gradients then synchronize across matching expert replicas.",
    fit:
      "Best when a MoE model needs both sparse expert capacity and higher data throughput from multiple expert-group replicas.",
    metrics: [
      ["Model state", "expert groups replicated"],
      ["Experts", "sharded inside each EP group"],
      ["Main traffic", "local all-to-all + expert grad sync"],
      ["Scaling limit", "expert load balance"],
    ],
  },
  hybrid: {
    label: "Hybrid 3D + Expert Layout",
    summary:
      "Large training jobs combine axes. A rank can belong to a DP group, TP group, PP stage, and EP group at once; each axis has its own process group and communication pattern.",
    communication:
      "TP collectives happen inside layers, PP sends activations between stages, EP routes MoE tokens, and DP/FSDP synchronizes the state shared across replicas.",
    fit:
      "Best for frontier-scale dense or MoE models where no single axis provides enough memory capacity or throughput.",
    metrics: [
      ["Example mesh", "DP=2, TP=2, PP=2, EP=2"],
      ["World size", "16 ranks in this diagram"],
      ["Main traffic", "collectives + p2p + all-to-all"],
      ["Scaling limit", "coordination and topology"],
    ],
  },
};

const parallelismPitfalls = {
  data:
    "DDP is easy to reason about, but it replicates optimizer state and gradients. Once the model no longer fits comfortably, increasing data parallel size alone will not solve memory pressure.",
  fsdp:
    "FSDP can trade memory for communication. If parameter all-gathers are not overlapped with compute, the job may fit but still scale poorly.",
  tensor:
    "Tensor parallelism is latency-sensitive because collectives happen inside layers. It usually belongs on the fastest local links, not across weak inter-node paths by default.",
  pipeline:
    "Pipeline parallelism can leave devices idle during fill and drain. Microbatch count, stage balance, and activation size decide whether the bubble is acceptable.",
  expert:
    "Expert parallelism depends on router balance. A theoretically sparse model can still stall if many tokens choose the same expert owner.",
  expertData:
    "EDP adds a second synchronization pattern: local expert routing plus matching-expert gradient sync across replicas. It needs process groups that reflect both axes.",
  hybrid:
    "Hybrid meshes are powerful but easy to mis-map. A bad rank order can place frequent TP collectives or expert all-to-all traffic on the slowest links.",
};

const state = {
  collective: "allreduce",
  algorithm: "ring",
  parallelism: "hybrid",
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
  collectiveWatch: document.querySelector("#collective-watch"),
  heroNetwork: document.querySelector("#hero-network"),
  topologySvg: document.querySelector("#topology-svg"),
  parallelismSelect: document.querySelector("#parallelism-select"),
  parallelismSvg: document.querySelector("#parallelism-svg"),
  parallelismTitle: document.querySelector("#parallelism-title"),
  parallelismSummary: document.querySelector("#parallelism-summary"),
  parallelismCommunication: document.querySelector("#parallelism-communication"),
  parallelismFit: document.querySelector("#parallelism-fit"),
  parallelismPitfall: document.querySelector("#parallelism-pitfall"),
  parallelismMetrics: document.querySelector("#parallelism-metrics"),
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

function addSvgTitle(node, text) {
  if (!text) return node;
  node.appendChild(createSvgElement("title", {}, text));
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

function rankPositions(n, cx = 500, cy = 335, radius = 252) {
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
  const offset = totalEdges > 18 ? ((index % 5) - 2) * 5 : ((index % 3) - 1) * 8;
  const ox = -uy * offset;
  const oy = ux * offset;
  const startX = from.x + ux * 54 + ox;
  const startY = from.y + uy * 54 + oy;
  const endX = to.x - ux * 54 + ox;
  const endY = to.y - uy * 54 + oy;
  const curve = totalEdges > 18 ? 24 : 34;
  const mx = (startX + endX) / 2 - uy * curve;
  const my = (startY + endY) / 2 + ux * curve;
  const path = `M ${startX} ${startY} Q ${mx} ${my} ${endX} ${endY}`;
  const phase = edge.phase === "reduce" ? "reduction" : edge.phase === "gather" ? "distribution" : "transfer";
  const packetStride = Math.max(1, Math.ceil(totalEdges / 12));
  const showPacket = totalEdges <= 16 || index % packetStride === 0;
  const flowGroup = addSvgTitle(
    createSvgElement("g", { class: "flow-group" }),
    `${edge.label}: active ${phase} edge for this phase.`,
  );
  flowGroup.appendChild(
    createSvgElement("path", {
      d: path,
      class: `edge active ${edge.phase || ""}`,
      markerEnd: edge.phase === "reduce" ? "url(#arrowhead-reduce)" : edge.phase === "gather" ? "url(#arrowhead-gather)" : "url(#arrowhead)",
    }),
  );
  if (showPacket) {
    const packet = createSvgElement("circle", {
      cx: (startX + endX) / 2,
      cy: (startY + endY) / 2,
      r: totalEdges > 18 ? 5.5 : 7.5,
      fill: chunkColors[(edge.from + edge.to) % chunkColors.length],
      class: "flow-packet",
    });
    flowGroup.appendChild(packet);
  }
  if (totalEdges <= 10) {
    const labelText = edge.phase === "reduce" ? "reduce" : edge.phase === "gather" ? "copy" : "send";
    const chipWidth = labelText.length * 7 + 20;
    const chipX = mx - chipWidth / 2;
    const chipY = my - 14;
    flowGroup.appendChild(
      createSvgElement("rect", {
        x: chipX,
        y: chipY,
        width: chipWidth,
        height: 20,
        rx: 10,
        class: "phase-chip-bg",
      }),
    );
    flowGroup.appendChild(
      createSvgElement("text", {
        x: mx,
        y: my + 1,
        "text-anchor": "middle",
        class: "phase-chip",
      }, labelText),
    );
  }
  svg.appendChild(flowGroup);
}

function renderRank(svg, position, rank, chunks, totalRanks) {
  const chunkList = chunks.length ? chunks.map((chunk) => `chunk ${chunk}`).join(", ") : "empty buffer";
  const g = addSvgTitle(
    createSvgElement("g", { class: "rank-node" }),
    `Rank ${rank}: currently holds ${chunkList}.`,
  );
  g.appendChild(createSvgElement("circle", { cx: position.x, cy: position.y, r: 42 }));
  g.appendChild(createSvgElement("text", { x: position.x, y: position.y - 7 }, `r${rank}`));
  g.appendChild(
    createSvgElement("text", { x: position.x, y: position.y + 18, class: "svg-small" }, `GPU ${rank}`),
  );

  const max = Math.max(totalRanks, 1);
  const barWidth = 84;
  const chunkWidth = Math.max(6, (barWidth - (max - 1) * 4) / max);
  const y = position.y + 58;
  const x = position.x - barWidth / 2;
  chunks.forEach((chunk) => {
    const rect = createSvgElement("rect", {
      x: x + chunk * (chunkWidth + 4),
      y,
      width: chunkWidth,
      height: 13,
      rx: 3,
      fill: chunkColors[chunk % chunkColors.length],
      class: "chunk",
    });
    addSvgTitle(rect, `Rank ${rank} buffer contains logical chunk ${chunk}.`);
    g.appendChild(rect);
  });
  g.appendChild(
    createSvgElement("rect", {
      x,
      y,
      width: barWidth,
      height: 13,
      rx: 3,
      fill: "none",
      stroke: "#b8c4c1",
      "stroke-width": 1,
    }),
  );
  if (chunks.length > 0) {
    g.appendChild(
      createSvgElement("rect", {
        x: position.x - 29,
        y: y + 19,
        width: 58,
        height: 18,
        rx: 9,
        class: "phase-chip-bg",
      }),
    );
    g.appendChild(createSvgElement("text", { x: position.x, y: y + 32, class: "buffer-label", "text-anchor": "middle" }, "buffer"));
  }
  svg.appendChild(g);
}

function renderLegend(svg, n) {
  const g = addSvgTitle(
    createSvgElement("g", { transform: "translate(18 42)", class: "chunk-legend" }),
    "Legend for logical tensor chunks shown in each rank buffer.",
  );
  g.appendChild(
    createSvgElement("rect", {
      x: -16,
      y: -28,
      width: 210,
      height: 174,
      rx: 10,
      fill: "#ffffff",
      opacity: 0.92,
      stroke: "#d7dfdc",
    }),
  );
  g.appendChild(createSvgElement("text", { x: 0, y: 0, class: "svg-label" }, "Chunk labels"));
  for (let i = 0; i < n; i++) {
    const col = Math.floor(i / 4);
    const row = i % 4;
    const x = col * 104;
    const y = 24 + row * 26;
    g.appendChild(
      createSvgElement("rect", {
        x,
        y,
        width: 20,
        height: 14,
        rx: 3,
        fill: chunkColors[i % chunkColors.length],
        class: "chunk",
      }),
    );
    g.appendChild(createSvgElement("text", { x: x + 28, y: y + 12, class: "svg-small" }, `chunk ${i}`));
  }
  g.appendChild(createSvgElement("text", { x: 0, y: 136, class: "svg-small" }, "Hover ranks, chunks, and flows."));
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
  [
    ["arrowhead", "#16817a"],
    ["arrowhead-reduce", "#c4544f"],
    ["arrowhead-gather", "#5b8c3a"],
  ].forEach(([id, fill]) => {
    const marker = createSvgElement("marker", {
      id,
      viewBox: "0 0 10 10",
      refX: "9",
      refY: "5",
      markerWidth: "7",
      markerHeight: "7",
      orient: "auto-start-reverse",
    });
    marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill }));
    defs.appendChild(marker);
  });
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
  els.collectiveWatch.textContent = collectiveWatch[state.collective];
  renderMetrics(def.metrics(state.ranks));
  renderTimeline(steps);
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

function renderTimeline(steps) {
  clear(els.timeline);
  for (let i = 0; i < steps.length; i++) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = i === state.step ? "active" : "";
    button.setAttribute("aria-label", `Go to step ${i + 1}`);
    button.title = steps[i].title;
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
    state.timer = setInterval(nextStep, 1850);
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

function parallelTiles() {
  const tileWidth = 158;
  const tileHeight = 74;
  const startX = 82;
  const startY = 142;
  const gapX = 238;
  const gapY = 132;
  return Array.from({ length: 16 }, (_, rank) => {
    const col = rank % 4;
    const row = Math.floor(rank / 4);
    const x = startX + col * gapX;
    const y = startY + row * gapY;
    return {
      rank,
      col,
      row,
      x,
      y,
      width: tileWidth,
      height: tileHeight,
      cx: x + tileWidth / 2,
      cy: y + tileHeight / 2,
    };
  });
}

function addParallelArrowDefs(svg) {
  const defs = createSvgElement("defs");
  const marker = createSvgElement("marker", {
    id: "parallel-arrow",
    viewBox: "0 0 10 10",
    refX: "9",
    refY: "5",
    markerWidth: "7",
    markerHeight: "7",
    orient: "auto",
  });
  marker.appendChild(createSvgElement("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#16817a" }));
  defs.appendChild(marker);
  svg.appendChild(defs);
}

function drawParallelGroup(svg, tiles, ranks, label, color, pad = 10) {
  const selected = ranks.map((rank) => tiles[rank]);
  const minX = Math.min(...selected.map((tile) => tile.x)) - pad;
  const minY = Math.min(...selected.map((tile) => tile.y)) - pad;
  const maxX = Math.max(...selected.map((tile) => tile.x + tile.width)) + pad;
  const maxY = Math.max(...selected.map((tile) => tile.y + tile.height)) + pad;
  const labelWidth = Math.min(maxX - minX - 24, Math.max(104, label.length * 6.7 + 24));
  const labelX = minX + 12;
  const labelY = minY - 12;
  const group = addSvgTitle(createSvgElement("g", { class: "parallel-group-wrap" }), label);
  group.appendChild(
    createSvgElement("rect", {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
      rx: 12,
      class: "parallel-group",
      stroke: color,
    }),
  );
  group.appendChild(
    createSvgElement("rect", {
      x: labelX - 8,
      y: labelY - 15,
      width: labelWidth,
      height: 23,
      rx: 8,
      class: "parallel-group-label-bg",
    }),
  );
  group.appendChild(createSvgElement("text", { x: labelX, y: labelY, class: "parallel-group-label", fill: color }, label));
  svg.appendChild(group);
}

function drawParallelTile(svg, tile, title, subtitle, color, fill = "#ffffff") {
  const g = addSvgTitle(
    createSvgElement("g", { class: "parallel-tile" }),
    `Rank ${tile.rank}: ${title}; ${subtitle}.`,
  );
  g.appendChild(
    createSvgElement("rect", {
      x: tile.x,
      y: tile.y,
      width: tile.width,
      height: tile.height,
      rx: 8,
      fill,
      stroke: color,
    }),
  );
  g.appendChild(
    createSvgElement("text", {
      x: tile.x + 16,
      y: tile.y + 19,
      "text-anchor": "start",
      "font-size": 12,
      "font-weight": 800,
      fill: color,
    }, `r${tile.rank}`),
  );
  g.appendChild(
    createSvgElement("text", {
      x: tile.cx,
      y: tile.y + 39,
      "font-size": 13,
      "font-weight": 800,
    }, title),
  );
  g.appendChild(createSvgElement("text", { x: tile.cx, y: tile.y + 58, class: "svg-small" }, subtitle));
  svg.appendChild(g);
}

function connectionPoint(from, to, outward = 1) {
  const dx = to.cx - from.cx;
  const dy = to.cy - from.cy;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const padX = from.width ? from.width / 2 + 12 : 22;
  const padY = from.height ? from.height / 2 + 12 : 22;
  const pad = Math.min(Math.abs(dx) > Math.abs(dy) ? padX : padY, 58);
  return {
    cx: from.cx + ux * pad * outward,
    cy: from.cy + uy * pad * outward,
  };
}

function drawParallelLine(svg, from, to, color = "#16817a", dashed = false, curve = 0, label = "traffic") {
  const start = connectionPoint(from, to, 1);
  const end = connectionPoint(to, from, 1);
  const d =
    curve === 0
      ? `M ${start.cx} ${start.cy} L ${end.cx} ${end.cy}`
      : `M ${start.cx} ${start.cy} Q ${(start.cx + end.cx) / 2} ${(start.cy + end.cy) / 2 - curve} ${end.cx} ${end.cy}`;
  const group = addSvgTitle(createSvgElement("g", { class: "parallel-flow-wrap" }), label);
  group.appendChild(
    createSvgElement("path", {
      d,
      class: `parallel-flow ${dashed ? "faint" : ""}`,
      stroke: color,
      "stroke-dasharray": dashed ? "8 8" : "",
      "marker-end": "url(#parallel-arrow)",
    }),
  );
  group.appendChild(
    createSvgElement("circle", {
      cx: (start.cx + end.cx) / 2,
      cy: (start.cy + end.cy) / 2,
      r: 5.5,
      fill: color,
      class: "flow-packet",
    }),
  );
  svg.appendChild(group);
}

function drawParallelLegend(svg, items) {
  const g = createSvgElement("g", { transform: "translate(82 650)" });
  items.forEach((item, index) => {
    const x = index * 210;
    g.appendChild(createSvgElement("rect", { x, y: 0, width: 18, height: 18, rx: 4, fill: item.color }));
    g.appendChild(createSvgElement("text", { x: x + 26, y: 14, class: "svg-small" }, item.label));
  });
  svg.appendChild(g);
}

function drawRingOnRanks(svg, tiles, ranks, color) {
  ranks.forEach((rank, index) => {
    drawParallelLine(
      svg,
      tiles[rank],
      tiles[ranks[(index + 1) % ranks.length]],
      color,
      true,
      index % 2 ? 26 : -26,
      `Rank ${rank} exchanges with rank ${ranks[(index + 1) % ranks.length]}.`,
    );
  });
}

function drawParallelismSvg(key) {
  const svg = els.parallelismSvg;
  const tiles = parallelTiles();
  clear(svg);
  addParallelArrowDefs(svg);

  svg.appendChild(createSvgElement("text", { x: 82, y: 54, class: "svg-label" }, parallelismDefinitions[key].label));
  svg.appendChild(
    createSvgElement("text", { x: 82, y: 82, class: "svg-small" }, "Sixteen ranks shown as a process-group mesh; colored boxes are communication or ownership groups."),
  );

  if (key === "data") {
    drawParallelGroup(svg, tiles, tiles.map((tile) => tile.rank), "DP group: full model replicas", "#16817a", 16);
    drawRingOnRanks(svg, tiles, tiles.map((tile) => tile.rank), "#16817a");
    tiles.forEach((tile) => drawParallelTile(svg, tile, "full model", `batch ${tile.rank}`, "#16817a", "#f8fbfa"));
    drawParallelLegend(svg, [
      { color: "#16817a", label: "gradient all-reduce" },
      { color: "#20252a", label: "replicated weights" },
    ]);
    return;
  }

  if (key === "fsdp") {
    drawParallelGroup(svg, tiles, tiles.map((tile) => tile.rank), "sharded data-parallel group", "#7758a6", 16);
    drawRingOnRanks(svg, tiles, tiles.map((tile) => tile.rank), "#7758a6");
    tiles.forEach((tile) => drawParallelTile(svg, tile, `param shard ${tile.rank % 8}`, "AG before compute", "#7758a6", "#fbf9ff"));
    drawParallelLegend(svg, [
      { color: "#7758a6", label: "all-gather / reduce-scatter" },
      { color: "#20252a", label: "logical DP semantics" },
    ]);
    return;
  }

  if (key === "tensor") {
    for (let row = 0; row < 4; row++) {
      const ranks = [0, 1, 2, 3].map((col) => row * 4 + col);
      drawParallelGroup(svg, tiles, ranks, `TP group ${row}: split one layer`, chunkColors[row], 10);
      ranks.slice(0, -1).forEach((rank) =>
        drawParallelLine(
          svg,
          tiles[rank],
          tiles[rank + 1],
          chunkColors[row],
          true,
          0,
          `Tensor-parallel exchange inside TP group ${row}: partial sums or activation shards move between rank ${rank} and rank ${rank + 1}.`,
        ),
      );
    }
    tiles.forEach((tile) => drawParallelTile(svg, tile, `W slice ${tile.col}`, `layer group ${tile.row}`, chunkColors[tile.row], "#f8fbfa"));
    drawParallelLegend(svg, [
      { color: "#16817a", label: "partial sums" },
      { color: "#c4544f", label: "row/column shards" },
      { color: "#7758a6", label: "same layer group" },
    ]);
    return;
  }

  if (key === "pipeline") {
    for (let col = 0; col < 4; col++) {
      const ranks = [0, 1, 2, 3].map((row) => row * 4 + col);
      drawParallelGroup(svg, tiles, ranks, `PP stage ${col}`, chunkColors[col], 10);
    }
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 3; col++) {
        drawParallelLine(
          svg,
          tiles[row * 4 + col],
          tiles[row * 4 + col + 1],
          "#bf7c21",
          false,
          0,
          `Pipeline microbatch lane ${row}: activations move from stage ${col} to stage ${col + 1}.`,
        );
      }
    }
    tiles.forEach((tile) => drawParallelTile(svg, tile, `layers ${tile.col}`, `microbatch lane ${tile.row}`, chunkColors[tile.col], "#fffaf3"));
    drawParallelLegend(svg, [
      { color: "#bf7c21", label: "activation send/recv" },
      { color: "#20252a", label: "depth sharded by stage" },
    ]);
    return;
  }

  if (key === "expert") {
    drawParallelGroup(svg, tiles, tiles.map((tile) => tile.rank), "EP group: experts distributed over ranks", "#c4544f", 16);
    const router = { cx: 520, cy: 102, width: 150, height: 46 };
    const routerGroup = addSvgTitle(createSvgElement("g"), "Router scores tokens, chooses experts, then dispatches token blocks to expert owners.");
    routerGroup.appendChild(createSvgElement("rect", { x: 445, y: 79, width: 150, height: 46, rx: 8, fill: "#273033" }));
    routerGroup.appendChild(createSvgElement("text", { x: 520, y: 108, "text-anchor": "middle", fill: "#fff", "font-weight": 760 }, "router"));
    svg.appendChild(routerGroup);
    [1, 3, 5, 8, 10, 12, 14].forEach((rank, index) => {
      drawParallelLine(
        svg,
        router,
        tiles[rank],
        chunkColors[index],
        false,
        38 + index * 2,
        `Token block ${index} is dispatched to expert owner rank ${rank}.`,
      );
    });
    tiles.forEach((tile) => drawParallelTile(svg, tile, `expert ${tile.rank % 8}`, "token owner", "#c4544f", "#fff8f8"));
    drawParallelLegend(svg, [
      { color: "#c4544f", label: "expert weights" },
      { color: "#16817a", label: "token dispatch" },
      { color: "#7758a6", label: "combine outputs" },
    ]);
    return;
  }

  if (key === "expertData") {
    const top = Array.from({ length: 8 }, (_, i) => i);
    const bottom = Array.from({ length: 8 }, (_, i) => i + 8);
    drawParallelGroup(svg, tiles, top, "EP group A / data replica 0", "#c4544f", 13);
    drawParallelGroup(svg, tiles, bottom, "EP group B / data replica 1", "#2f7fa4", 13);
    for (let i = 0; i < 8; i++) {
      drawParallelLine(
        svg,
        tiles[i],
        tiles[i + 8],
        "#5b8c3a",
        true,
        0,
        `Matching expert ${i} synchronizes gradients across data replicas.`,
      );
    }
    [...top, ...bottom].forEach((rank) => {
      const tile = tiles[rank];
      const color = rank < 8 ? "#c4544f" : "#2f7fa4";
      drawParallelTile(svg, tile, `expert ${rank % 8}`, `replica ${rank < 8 ? 0 : 1}`, color, rank < 8 ? "#fff8f8" : "#f5fbff");
    });
    drawParallelLegend(svg, [
      { color: "#c4544f", label: "expert group A" },
      { color: "#2f7fa4", label: "expert group B" },
      { color: "#5b8c3a", label: "matching expert grad sync" },
    ]);
    return;
  }

  const topReplica = Array.from({ length: 8 }, (_, i) => i);
  const bottomReplica = Array.from({ length: 8 }, (_, i) => i + 8);
  drawParallelGroup(svg, tiles, topReplica, "DP replica 0: TP x PP x EP submesh", "#16817a", 15);
  drawParallelGroup(svg, tiles, bottomReplica, "DP replica 1: matching submesh", "#7758a6", 15);
  [0, 1, 2, 3].forEach((col) => {
    drawParallelLine(svg, tiles[col], tiles[col + 4], "#bf7c21", false, 0, `Pipeline activation path inside data replica 0, column ${col}.`);
    drawParallelLine(svg, tiles[col + 8], tiles[col + 12], "#bf7c21", false, 0, `Pipeline activation path inside data replica 1, column ${col}.`);
  });
  [0, 2, 8, 10].forEach((rank) =>
    drawParallelLine(svg, tiles[rank], tiles[rank + 1], "#2f7fa4", true, 0, `Tensor-parallel collective between rank ${rank} and rank ${rank + 1}.`),
  );
  [4, 6, 12, 14].forEach((rank) =>
    drawParallelLine(svg, tiles[rank], tiles[rank + 1], "#c4544f", true, 0, `Expert-parallel token path between rank ${rank} and rank ${rank + 1}.`),
  );
  tiles.forEach((tile) => {
    const dp = tile.row < 2 ? 0 : 1;
    const pp = tile.row % 2;
    const tp = tile.col % 2;
    const ep = tile.col < 2 ? 0 : 1;
    drawParallelTile(svg, tile, `DP${dp} PP${pp}`, `TP${tp} EP${ep}`, dp === 0 ? "#16817a" : "#7758a6", "#f9fbfa");
  });
  drawParallelLegend(svg, [
    { color: "#16817a", label: "data replica" },
    { color: "#2f7fa4", label: "TP collectives" },
    { color: "#bf7c21", label: "PP activation path" },
    { color: "#c4544f", label: "EP token path" },
  ]);
}

function renderParallelismMetrics(metrics) {
  clear(els.parallelismMetrics);
  metrics.forEach(([label, value]) => {
    const metric = document.createElement("div");
    metric.className = "metric";
    const span = document.createElement("span");
    span.textContent = label;
    const strong = document.createElement("strong");
    strong.textContent = value;
    metric.append(span, strong);
    els.parallelismMetrics.appendChild(metric);
  });
}

function renderParallelism() {
  if (!els.parallelismSvg) return;
  const def = parallelismDefinitions[state.parallelism];
  els.parallelismTitle.textContent = def.label;
  els.parallelismSummary.textContent = def.summary;
  els.parallelismCommunication.textContent = def.communication;
  els.parallelismFit.textContent = def.fit;
  els.parallelismPitfall.textContent = parallelismPitfalls[state.parallelism];
  renderParallelismMetrics(def.metrics);
  drawParallelismSvg(state.parallelism);
}

function initParallelismControls() {
  if (!els.parallelismSelect) return;
  Object.entries(parallelismDefinitions).forEach(([value, def]) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = def.label;
    els.parallelismSelect.appendChild(option);
  });
  els.parallelismSelect.value = state.parallelism;
  els.parallelismSelect.addEventListener("change", () => {
    state.parallelism = els.parallelismSelect.value;
    renderParallelism();
  });
}

initControls();
initParallelismControls();
renderHeroNetwork();
renderTopology();
renderParallelism();
renderSimulator();
