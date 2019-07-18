from .edge import StateEdge
from .node import StateNode, expand_anchors, compress_name
from ..constraints.amr import NAME
from ..constraints.validation import ROOT_ID, ROOT_LAB, ANCHOR_LAB
from ..recategorization import resolve


class RefGraph:
    def __init__(self, graph, conllu, framework):
        """
        Create reference graph, which is a copy of graph but has StateNodes and StateEdges instead of Nodes and Edges.
        Other differences:
        (1) Virtual root to support top nodes (children of the root will be marked as top=True)
        (2) Virtual terminals to support anchors
        (3) Strings in node labels are replaced with placeholder when they match aligned terminal text
        :return: RefGraph with nodes, edges, root and terminals
        """
        self.framework = framework
        self.terminals = [StateNode(i, conllu_node.id, text=conllu_node.label,
                                    anchors=expand_anchors(conllu_node.anchors),
                                    properties=dict(zip(conllu_node.properties or (), conllu_node.values or ())))
                          for i, conllu_node in enumerate(conllu.nodes)]  # Virtual node for tokens
        self.root = StateNode(ROOT_ID, ROOT_ID, is_root=True)  # Virtual root for tops
        self.nodes = [self.root]
        id2node = {}
        offset = len(conllu.nodes) + 1
        self.non_virtual_nodes = []
        self.edges = []
        for graph_node in graph.nodes:
            node_id = graph_node.id + offset
            id2node[node_id] = node = \
                StateNode(node_id, node_id, ref_node=graph_node, label=graph_node.label,
                          anchors=expand_anchors(graph_node.anchors),
                          properties=dict(zip(graph_node.properties or (), graph_node.values or ())))
            self.nodes.append(node)
            self.non_virtual_nodes.append(node)
            if graph_node.is_top:
                self.edges.append(StateEdge(self.root, node, ROOT_LAB).add())
            if node.anchors:
                for terminal in self.terminals:
                    if node.anchors & terminal.anchors:
                        self.edges.append(StateEdge(node, terminal, ANCHOR_LAB).add())
        for edge in graph.edges:
            self.edges.append(StateEdge(id2node[edge.src + offset],
                                        id2node[edge.tgt + offset], edge.lab,
                                        dict(zip(edge.attributes or (), edge.values or ()))).add())
        for node in self.non_virtual_nodes:
            node.label = resolve(node, node.label, introduce_placeholders=True)
            if node.properties:
                if self.framework == "amr" and node.label == NAME:
                    node.properties = compress_name(node.properties)
                node.properties = {prop: resolve(node, value, introduce_placeholders=True)
                                   for prop, value in node.properties.items()}
