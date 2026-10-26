"""Live cross-group isolation tests for the MCP server against FalkorDB + a real LLM.

FalkorDB stores each group_id in its own graph. One MCP server process serves every
group, and its queue runs one worker per group, so writes to different groups run
concurrently through a single shared Graphiti client. Older graphiti-core rebound
that client's shared driver to the target graph inside add_episode; with an await
in between, a concurrent write for another group could land in the wrong graph or
see the other group's entities during deduplication.

These tests write two groups concurrently through one server, then read each graph
directly and assert it holds only its own group's episodes, entities and edges.
Prerequisites and skipping are the same as test_live_falkordb_int.py.
"""

import asyncio
import time
from contextlib import suppress

from falkordb import FalkorDB

from tests.test_live_falkordb_int import (
    FALKORDB_URI,
    LiveMCPClient,
    _raise_on_error,
    _unique_group_id,
)

# Disjoint, distinctive vocabularies so any leak across graphs is detectable by name.
GROUP_A_EPISODES = [
    'Zephyrine Okonkwo is the CEO of Quillmark Robotics, based in Tallinn.',
    'Quillmark Robotics acquired Brindlewood Sensors in a deal led by Zephyrine Okonkwo.',
    'Ottoline Marsh is the CFO of Quillmark Robotics and reports to Zephyrine Okonkwo.',
]
GROUP_B_EPISODES = [
    'Bartholomew Vance founded Tessaract Farms, an orchard business in Hobart.',
    'Tessaract Farms partnered with Calloway Mills to export cider made by Bartholomew Vance.',
    'Persephone Lund is the head grower at Tessaract Farms and works with Bartholomew Vance.',
]
GROUP_A_MARKERS = {'zephyrine', 'okonkwo', 'quillmark', 'brindlewood', 'ottoline', 'tallinn'}
GROUP_B_MARKERS = {'bartholomew', 'vance', 'tessaract', 'calloway', 'persephone', 'hobart'}

GROUP_A_TRIPLETS = [
    (
        'Ignatius Quillmark',
        'ADVISES',
        'Ignatius Quillmark advises Brindlewood Sensors',
        'Brindlewood Sensors',
    ),
    ('Ottoline Marsh', 'MENTORS', 'Ottoline Marsh mentors Zephyrine Okonkwo', 'Zephyrine Okonkwo'),
]
GROUP_B_TRIPLETS = [
    ('Calloway Mills', 'SUPPLIES', 'Calloway Mills supplies Tessaract Farms', 'Tessaract Farms'),
    ('Persephone Lund', 'TRAINS', 'Persephone Lund trains Bartholomew Vance', 'Bartholomew Vance'),
]


def _graph_query(db: FalkorDB, graph: str, cypher: str) -> list[list]:
    if graph not in db.list_graphs():
        return []
    return db.select_graph(graph).ro_query(cypher).result_set


def _census(db: FalkorDB, groups: list[str]) -> dict[str, dict[str, int]]:
    """Where each test group's nodes live: {graph: {group_id: node count}}."""
    census = {}
    for graph in db.list_graphs():
        rows = _graph_query(
            db, graph, f'MATCH (n) WHERE n.group_id IN {groups!r} RETURN n.group_id, count(n)'
        )
        if rows:
            census[graph] = {r[0]: r[1] for r in rows}
    return census


def _assert_graph_isolated(
    db: FalkorDB, graph: str, other_graph: str, own: set[str], other: set[str]
) -> None:
    node_groups = {r[0] for r in _graph_query(db, graph, 'MATCH (n) RETURN DISTINCT n.group_id')}
    assert node_groups == {graph}, (
        f'graph {graph} holds nodes of groups {node_groups}; '
        f'census: {_census(db, [graph, other_graph])}'
    )

    edge_groups = {
        r[0] for r in _graph_query(db, graph, 'MATCH ()-[e]->() RETURN DISTINCT e.group_id')
    }
    assert edge_groups <= {graph}, f'graph {graph} holds edges of groups {edge_groups}'

    texts = [
        str(r[0]).lower()
        for cypher in (
            'MATCH (n:Entity) RETURN n.name',
            'MATCH (n:Episodic) RETURN n.content',
            'MATCH ()-[e:RELATES_TO]->() RETURN e.fact',
        )
        for r in _graph_query(db, graph, cypher)
    ]
    leaked = sorted({m for m in other for t in texts if m in t})
    assert not leaked, f"graph {graph} contains the other group's terms: {leaked}"
    found = {m for m in own for t in texts if m in t}
    assert found, f'graph {graph} contains none of its own terms; texts={texts}'


def _assert_default_graph_clean(db: FalkorDB, groups: list[str]) -> None:
    for graph in db.list_graphs():
        if graph in groups:
            continue
        stray = _graph_query(db, graph, f'MATCH (n) WHERE n.group_id IN {groups!r} RETURN count(n)')
        assert not stray or stray[0][0] == 0, f'graph {graph} holds {stray[0][0]} test nodes'


async def _wait_for_episode_count(db: FalkorDB, groups: list[str], expected: int) -> int:
    """Poll every graph until `expected` test episodes exist anywhere (or time out)."""
    deadline = time.monotonic() + 600
    total = 0
    while time.monotonic() < deadline:
        total = sum(
            _graph_query(
                db, graph, f'MATCH (e:Episodic) WHERE e.group_id IN {groups!r} RETURN count(e)'
            )[0][0]
            for graph in db.list_graphs()
        )
        if total >= expected:
            return total
        await asyncio.sleep(5)
    return total


async def test_concurrent_add_memory_two_groups_stay_isolated():
    # Group A is the server's default group and group B is passed explicitly,
    # mirroring production (gcc_deals default, gcc_user_<name> explicit).
    group_a, group_b = _unique_group_id() + 'a', _unique_group_id() + 'b'
    db = FalkorDB.from_url(FALKORDB_URI)
    async with LiveMCPClient(group_a) as client:
        try:
            calls = []
            for i, (body_a, body_b) in enumerate(
                zip(GROUP_A_EPISODES, GROUP_B_EPISODES, strict=True)
            ):
                for group, body in ((group_a, body_a), (group_b, body_b)):
                    calls.append(
                        client.call(
                            'add_memory',
                            {
                                'name': f'{group} episode {i}',
                                'episode_body': body,
                                'source': 'text',
                                'source_description': 'group isolation test',
                                'group_id': group,
                            },
                        )
                    )
            for resp in await asyncio.gather(*calls):
                _raise_on_error('add_memory', resp)

            expected = len(GROUP_A_EPISODES) + len(GROUP_B_EPISODES)
            total = await _wait_for_episode_count(db, [group_a, group_b], expected)
            assert total == expected, f'{total}/{expected} episodes processed in time'

            _assert_graph_isolated(db, group_a, group_b, GROUP_A_MARKERS, GROUP_B_MARKERS)
            _assert_graph_isolated(db, group_b, group_a, GROUP_B_MARKERS, GROUP_A_MARKERS)
            _assert_default_graph_clean(db, [group_a, group_b])
            for group, n in ((group_a, len(GROUP_A_EPISODES)), (group_b, len(GROUP_B_EPISODES))):
                count = _graph_query(db, group, 'MATCH (e:Episodic) RETURN count(e)')[0][0]
                assert count == n, f'graph {group} has {count} episodes, expected {n}'
        finally:
            for group in (group_a, group_b):
                with suppress(Exception):
                    await client.call('clear_graph', {'group_ids': [group]})


async def test_concurrent_add_triplet_two_groups_stay_isolated():
    group_a, group_b = _unique_group_id() + 'a', _unique_group_id() + 'b'
    db = FalkorDB.from_url(FALKORDB_URI)
    async with LiveMCPClient(group_a) as client:
        try:
            calls = []
            for triplet_a, triplet_b in zip(GROUP_A_TRIPLETS, GROUP_B_TRIPLETS, strict=True):
                for group, (source, edge_name, fact, target) in (
                    (group_a, triplet_a),
                    (group_b, triplet_b),
                ):
                    calls.append(
                        client.call(
                            'add_triplet',
                            {
                                'source_node_name': source,
                                'edge_name': edge_name,
                                'fact': fact,
                                'target_node_name': target,
                                'group_id': group,
                            },
                        )
                    )
            for resp in await asyncio.gather(*calls):
                _raise_on_error('add_triplet', resp)

            _assert_graph_isolated(db, group_a, group_b, GROUP_A_MARKERS, GROUP_B_MARKERS)
            _assert_graph_isolated(db, group_b, group_a, GROUP_B_MARKERS, GROUP_A_MARKERS)
            _assert_default_graph_clean(db, [group_a, group_b])
            for group, n in ((group_a, len(GROUP_A_TRIPLETS)), (group_b, len(GROUP_B_TRIPLETS))):
                count = _graph_query(db, group, 'MATCH ()-[e:RELATES_TO]->() RETURN count(e)')[0][0]
                assert count == n, f'graph {group} has {count} facts, expected {n}'
        finally:
            for group in (group_a, group_b):
                with suppress(Exception):
                    await client.call('clear_graph', {'group_ids': [group]})
