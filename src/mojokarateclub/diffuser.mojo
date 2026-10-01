"""Eulerian diffusions, the PRNG-driven growth that feeds them, and the
Hierholzer tour that linearises it.

Port of `karateclub/utils/diffuser.py`. Upstream grows a random connected
subgraph around a source node and then hands it to `networkx.eulerian_circuit`,
which `Diff2Vec` feeds to `Word2Vec` as if it were a walk. A Mojo kernel holds
no `networkx` graph, so the graph arrives as a CSR, the `nx.DiGraph` the
subgraph lives in arrives as an insertion-ordered adjacency list in the
caller's scratch, and the tour comes back as a flat `Int32` that the caller
formats with `str(...)`, as upstream's `[str(u) for u, v in ...]` does.

Two representations have to agree exactly, because upstream's tour depends on
them. `nx.DiGraph` iterates a node's out-edges in *insertion* order, and
`arbitrary_element(edges(current_vertex))` takes the first of them, so this
module appends to the tail of each node's list rather than the head. And
`eulerian_circuit` reverses a `DiGraph` before the search, which for this
subgraph - every edge added in both directions - is the identity, so the
search runs on the same edge set upstream searches.

The draws are the usual divergence: `random.sample` and `random.choice` are
CPython's `_random`, and `walker` carries MT19937 in the same place instead.

Nothing allocates. The infected list, the subgraph, the stack and the tour are
the caller's.
"""

import walker

comptime IPtr = Pointer[Int32, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]


def sub_head(head: IPtr, stamp: IPtr, node: Int, generation: Int) -> Int32:
    """The first out-edge of `node` in the subgraph, or -1.

    -1 covers both "no out-edge" and "not in the subgraph", which is the
    distinction the search does not need. The stamp is what makes the search
    safe across diffusions without clearing the subgraph: a node the previous
    diffusion touched reads as empty because its stamp is stale.
    """
    if stamp.unsafe_load(node) != Int32(generation):
        return Int32(-1)
    return head.unsafe_load(node)


def sub_add_edge(
    head: IPtr,
    tail: IPtr,
    degree: IPtr,
    stamp: IPtr,
    next_edge: IPtr,
    edge_dst: IPtr,
    generation: Int,
    source: Int,
    target: Int,
    edge_count: Int,
) -> Int:
    """`sub_graph.add_edge(source, target)`, appended to the insertion-ordered
    list, returning the next free edge slot."""
    if stamp.unsafe_load(source) != Int32(generation):
        head.unsafe_offset(source)[] = Int32(-1)
        tail.unsafe_offset(source)[] = Int32(-1)
        degree.unsafe_offset(source)[] = Int32(0)
        stamp.unsafe_offset(source)[] = Int32(generation)
    var last = tail.unsafe_load(source)
    next_edge.unsafe_offset(edge_count)[] = Int32(-1)
    edge_dst.unsafe_offset(edge_count)[] = Int32(target)
    if last == Int32(-1):
        head.unsafe_offset(source)[] = Int32(edge_count)
    else:
        next_edge.unsafe_offset(Int(last))[] = Int32(edge_count)
    tail.unsafe_offset(source)[] = Int32(edge_count)
    degree.unsafe_offset(source)[] = degree.unsafe_load(source) + Int32(1)
    return edge_count + 1


def eulerian_circuit(
    head: IPtr,
    degree: IPtr,
    stamp: IPtr,
    next_edge: IPtr,
    edge_dst: IPtr,
    generation: Int,
    source: Int,
    stack: IPtr,
    circuit: IPtr,
) -> Int:
    """`_simplegraph_eulerian_circuit` from `networkx/algorithms/euler.py`.

    The stack search, in upstream's structure: push while the top vertex still
    has an out-edge, taking the first in insertion order and removing it, and
    on a dead end pop and emit `(last_vertex, current_vertex)`. The first pop
    has no `last_vertex` and is not emitted, and there is one more pop than
    push, so the caller's `[str(u) for u, v in ...]` has one id per edge.

    The only substitution is `head[vertex] == -1` for `degree(vertex) == 0`:
    the head sentinel and the degree are the same fact, because the list is
    only ever consumed from the head.

    `stack` holds one more id than the subgraph has edges and `circuit` exactly
    as many; both are the caller's.
    """
    var stacktop = 0
    stack.unsafe_offset(0)[] = Int32(source)
    var last_vertex = -1
    var count = 0
    while stacktop >= 0:
        var current = Int(stack.unsafe_load(stacktop))
        var edge = sub_head(head, stamp, current, generation)
        if edge == Int32(-1):
            if last_vertex != -1:
                circuit.unsafe_offset(count)[] = Int32(last_vertex)
                count += 1
            last_vertex = current
            stacktop -= 1
        else:
            head.unsafe_offset(current)[] = next_edge.unsafe_load(Int(edge))
            degree.unsafe_offset(current)[] = degree.unsafe_load(current) - Int32(1)
            stacktop += 1
            stack.unsafe_offset(stacktop)[] = edge_dst.unsafe_load(Int(edge))
    return count


def frontier(
    indptr: IPtr, indices: IPtr, marked: IPtr, node: Int, generation: Int
) -> Tuple[Int, Int]:
    """`(outside, inside)` for `node` in the growth.

    `marked[u]` is the generation `u` was infected in, so a neighbour is
    outside the set exactly when its mark is stale. A node's own self loop is
    neither: `_ensure_integrity` gives every node one and the growth never
    crosses it. The caller adds the difference to its boundary count, so this
    must be read before `node` is marked if `node` may already be infected.
    """
    var start = Int(indptr.unsafe_load(node))
    var stop = Int(indptr.unsafe_load(node + 1))
    var outside = 0
    var inside = 0
    var i = start
    while i < stop:
        var neighbour = Int(indices.unsafe_load(i))
        if neighbour != node:
            if marked.unsafe_load(neighbour) != Int32(generation):
                outside += 1
            else:
                inside += 1
        i += 1
    return (outside, inside)


def eulerian_diffuser_run_diffusion_process(
    indptr: IPtr,
    indices: IPtr,
    node: Int,
    diffusion_cover: Int,
    generation: Int,
    infected: IPtr,
    marked: IPtr,
    head: IPtr,
    tail: IPtr,
    degree: IPtr,
    stamp: IPtr,
    next_edge: IPtr,
    edge_dst: IPtr,
    stack: IPtr,
    circuit: IPtr,
    state: U32Ptr,
) -> Int:
    """`_run_diffusion_process(node)`: grow the subgraph around `node` and
    return the length of the Eulerian tour written into `circuit`.

    `infected[0:counter]` is the infection list in the order upstream appends
    to it, `circuit` receives the tour, and the rest is subgraph scratch. A
    tour is as long as the edges added, at most `2 * (diffusion_cover - 1)`.

    The graph must have at least one neighbour at `end_point` on every step:
    upstream's `random.choice(nebs)` raises `IndexError` on an empty list, and
    a kernel cannot raise it. `Estimator._ensure_integrity` gives every node a
    self loop, so the estimators never reach that, and the Python layer checks
    for it before calling.

    The loop carries one condition upstream does not have, and it is the
    condition under which upstream cannot terminate: `boundary` counts the
    edges between the infected set and the rest of the graph, and a diffusion
    whose set has no such edge left can never infect anything more, so the
    growth stops there. Upstream's `while infected_counter < diffusion_cover`
    spins forever on any graph where the cover exceeds the source's component,
    which is every graph the moment `diffusion_cover > n`. Whenever a
    boundary edge does exist the walk can still find it, so this only changes
    the outcome where upstream hangs.
    """
    infected.unsafe_offset(0)[] = Int32(node)
    var counts = frontier(indptr, indices, marked, node, generation)
    marked.unsafe_offset(node)[] = Int32(generation)
    var boundary = counts[0]
    var infected_counter = 1
    var edge_count = 0
    while infected_counter < diffusion_cover and boundary > 0:
        # `random.sample(infected, 1)[0]`: at k == 1 the pool swap has no
        # second pick to hide behind, so this is one `randbelow` over the
        # list as it stands.
        var end_point = Int(walker.sample_one(state, infected, 0, infected_counter))
        # `random.choice(nebs)` is `seq[seq._randbelow(len(seq))]`.
        var sample = Int(
            walker.sample_one(
                state,
                indices,
                Int(indptr.unsafe_load(end_point)),
                Int(indptr.unsafe_load(end_point + 1))
                - Int(indptr.unsafe_load(end_point)),
            )
        )
        # `sample not in infected`, which upstream answers with
        # `list.__contains__` over the list it has been appending to.
        var seen = False
        var i = 0
        while i < infected_counter:
            if Int(infected.unsafe_load(i)) == sample:
                seen = True
            i += 1
        if not seen:
            infected_counter += 1
            infected.unsafe_offset(infected_counter - 1)[] = Int32(sample)
            # Every edge from the sample into the set stops crossing the
            # boundary and every edge out of it starts, and the self loop
            # `frontier` drops is not one of them.
            counts = frontier(indptr, indices, marked, sample, generation)
            marked.unsafe_offset(sample)[] = Int32(generation)
            boundary += counts[0] - counts[1]
            # `sub_graph.add_edges_from([(end_point, sample), (sample, end_point)])`
            edge_count = sub_add_edge(
                head,
                tail,
                degree,
                stamp,
                next_edge,
                edge_dst,
                generation,
                end_point,
                sample,
                edge_count,
            )
            edge_count = sub_add_edge(
                head,
                tail,
                degree,
                stamp,
                next_edge,
                edge_dst,
                generation,
                sample,
                end_point,
                edge_count,
            )
            if infected_counter == diffusion_cover:
                break
    return eulerian_circuit(
        head,
        degree,
        stamp,
        next_edge,
        edge_dst,
        generation,
        node,
        stack,
        circuit,
    )


def eulerian_diffuser_do_diffusions(
    indptr: IPtr,
    indices: IPtr,
    n: Int,
    diffusion_number: Int,
    diffusion_cover: Int,
    diffusions: IPtr,
    offsets: IPtr,
    state: U32Ptr,
    infected: IPtr,
    marked: IPtr,
    head: IPtr,
    tail: IPtr,
    degree: IPtr,
    stamp: IPtr,
    next_edge: IPtr,
    edge_dst: IPtr,
    stack: IPtr,
) -> Int:
    """`do_diffusions(graph)`: `diffusion_number` diffusions from every node,
    in node order then diffusion order, concatenated into the caller's
    `diffusions` with a start offset per diffusion in `offsets`.

    Returns the total number of ids written. One diffusion is one tour, so
    `offsets` has `n * diffusion_number + 1` entries and its last entry is the
    return value. The generation stamp advances per diffusion instead of the
    subgraph being cleared, which is what makes the node-order loop safe and
    doubles as the infected-set mark.

    `diffusions` must hold `n * diffusion_number * (2 * diffusion_cover - 2)`
    ids, the longest a tour can be. Each tour is written straight into it at
    the running total, so there is no separate per-tour buffer to size.
    """
    var total = 0
    var count = 0
    var generation = 0
    offsets.unsafe_offset(0)[] = Int32(0)
    for node in range(n):
        for _ in range(diffusion_number):
            generation += 1
            var written = eulerian_diffuser_run_diffusion_process(
                indptr,
                indices,
                node,
                diffusion_cover,
                generation,
                infected,
                marked,
                head,
                tail,
                degree,
                stamp,
                next_edge,
                edge_dst,
                stack,
                diffusions.unsafe_offset(total),
                state,
            )
            total += written
            count += 1
            offsets.unsafe_offset(count)[] = Int32(total)
    return total
