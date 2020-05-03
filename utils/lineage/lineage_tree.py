"""
    Content: This file defines tools for tracking model lineage in arena
    Author: Yiyuan Yang
    Date: April. 29th 2020
"""


class LineageNode(object):
    def __init__(
        self,
        model_id,
        left_lineage_node=None,
        right_lineage_node=None
    ):
        self.model_id = model_id
        self.left = left_lineage_node
        self.right = right_lineage_node

    def __str__(self):
        return str(self.model_id)


class Lineage(object):
    def __init__(self, model_id, left_lineage, right_lineage):
        left = self.copy_lineage(left_lineage)
        right = self.copy_lineage(right_lineage)
        self.root = LineageNode(model_id, left, right)

    def copy_lineage(self, lineage):
        if lineage is None:
            return None
        root = LineageNode(model_id=lineage.model_id())
        root.left_parent = self.copy_lineage(lineage.left())
        root.right_parent = self.copy_lineage(lineage.right())
        return root

    def model_id(self):
        return self.root.model_id

    def left(self):
        return self.root.left

    def right(self):
        return self.root.right

    def traverse(self):
        lineage_string = "\n"
        current_level = [self.root]
        while current_level:
            lineage_string += \
                ' '.join(str(node) for node in current_level) + "\n"
            next_level = list()
            for n in current_level:
                if n.left:
                    next_level.append(n.left)
                if n.right:
                    next_level.append(n.right)
            current_level = next_level
        return lineage_string

    def __str__(self):
        return self.traverse()
