from __future__ import annotations
import copy
from .templates import TEMPLATES


class Builder:
    def __init__(self):
        self.nodes, self.links, self.groups = [], [], []
        self._nid = self._lid = 0

    def add(self, ntype, widgets=None, pos=(0, 0), title=None, mode=0, color=None, bgcolor=None):
        try:
            n = copy.deepcopy(TEMPLATES[ntype])
        except KeyError:
            raise KeyError(
                f"no node template for {ntype!r} — add it to EXTRA_TEMPLATES in templates.py, "
                f"or ensure a node of that type exists in the harvest source (MainGraphv10.json)"
            ) from None
        self._nid += 1
        n["id"] = self._nid
        n["flags"] = {}
        n["mode"] = mode
        n["pos"] = [float(pos[0]), float(pos[1])]
        n["order"] = 0
        for inp in n.get("inputs", []):
            inp["link"] = None
        for out in n.get("outputs", []):
            out["links"] = []
        if widgets is not None:
            n["widgets_values"] = widgets
        if title is not None:
            n["title"] = title
        elif "title" in n:
            del n["title"]
        if color:
            n["color"] = color
        if bgcolor:
            n["bgcolor"] = bgcolor
        self.nodes.append(n)
        return n["id"]

    def _node(self, nid):
        for n in self.nodes:
            if n["id"] == nid:
                return n
        raise KeyError(nid)

    @staticmethod
    def _slot(node, key, name):
        for i, s in enumerate(node[key]):
            if s["name"] == name:
                return i, s
        raise KeyError(f"node {node['id']} ({node['type']}) has no {key[:-1]} {name!r}; "
                       f"have {[s['name'] for s in node[key]]}")

    def link(self, src_id, out_name, dst_id, in_name):
        s, d = self._node(src_id), self._node(dst_id)
        oi, oslot = self._slot(s, "outputs", out_name)
        ii, islot = self._slot(d, "inputs", in_name)
        self._lid += 1
        self.links.append([self._lid, src_id, oi, dst_id, ii, islot.get("type") or oslot.get("type")])
        oslot["links"].append(self._lid)
        islot["link"] = self._lid
        return self._lid

    def group(self, title, node_ids, color="#3f789e"):
        node_ids = [i for i in node_ids if i is not None]
        xs, ys, x2, y2 = [], [], [], []
        for nid in node_ids:
            n = self._node(nid)
            x, y = n["pos"]; w, h = n.get("size", [300, 120])
            xs.append(x); ys.append(y); x2.append(x + w); y2.append(y + h)
        bx, by = min(xs) - 40, min(ys) - 70
        self.groups.append({"title": title, "bounding": [bx, by, max(x2) - bx + 40, max(y2) - by + 40],
                            "color": color, "font_size": 24, "flags": {}})

    def _order(self):
        from collections import deque
        succ = {n["id"]: [] for n in self.nodes}
        indeg = {n["id"]: 0 for n in self.nodes}
        for _, sf, _, st, _, _ in self.links:
            succ[sf].append(st); indeg[st] += 1
        q = deque(sorted(i for i, d in indeg.items() if d == 0))
        order, seen = [], set()
        while q:
            i = q.popleft()
            if i in seen:
                continue
            seen.add(i); order.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
        for n in self.nodes:
            if n["id"] not in seen:
                order.append(n["id"])
        for idx, nid in enumerate(order):
            self._node(nid)["order"] = idx

    def build(self, revision=1):
        self._order()
        return {"id": "", "revision": revision, "last_node_id": self._nid, "last_link_id": self._lid,
                "nodes": self.nodes, "links": self.links, "groups": self.groups,
                "config": {}, "extra": {}, "version": 0.4}
