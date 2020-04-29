"""
    Content: This file defines tools for tracking model lineage in arena
    Author: Yiyuan Yang
    Date: April. 29th 2020
"""


class LineageNode(object):
    def __init__(self, model_id, left_parent_tree=None, right_parent_tree=None):
        """
            Either model_id only, or all three arguments are present
        """
        assert model_id is not None and (
            (left_parent_tree is None and right_parent_tree is None) or 
            (left_parent_tree is not None and right_parent_tree is not None)
        ), "Either Provide model ID only, or All three arguements in LineageNode"
        self.model_id = model_id
        if left_parent_tree is not None:
            self.lineage_length = max(left_parent_tree)
        
        
    def copy_lineage(self, lineage):
        if lineage is None:
            return None
        root = LineageNode(model_id=lineage.model_id)
        root.lineage_length = lineage.lineage_length
        root.left_parent = self.copy_lineage(lineage.left_parent)
        root.right_parent = self.copy_lineage(lineage.right_parent)
        return root

    
