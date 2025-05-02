"""
Box counting algorithm and octotree class object.

"""

import numpy as np


class box:
    
    def __init__(self, i, xlims, ylims, zlims, x_points):
        """
        Args:
            i (float): Box nesting level
            xlims (list): [lower x bound, uppper x bound]
            ylims (list): [lower y bound, uppper y bound]
            zlims (list): [lower z bound, uppper z bound]
            x_points (np.ndarray): Data set

        """
        
        ### Box level
        self.i = i
        
        ### Box domain
        self.xlims = xlims
        self.ylims = ylims
        self.zlims = zlims
        
        ### Points contained within box
        self.x_points = x_points
        
        ### Child data
        self.children = []

        
    @staticmethod
    def valid_points(x_data, xlims, ylims, zlims):
        """
        Helper to find nested points.
        
        Args:
            x_data (np.ndarray): Data set
            xlims (list): [lower x bound, uppper x bound]
            ylims (list): [lower y bound, uppper y bound]
            zlims (list): [lower z bound, uppper z bound]
            x_points (np.ndarray): Dataset
            
        Returns:
            np.ndarray: Data points contained within bounds
        """
        
        ### Create mask for points within bounds
        mask = (
            (x_data[:,0] >= xlims[0]) & (x_data[:,0] < xlims[1]) &
            (x_data[:,1] >= ylims[0]) & (x_data[:,1] < ylims[1]) &
            (x_data[:,2] >= zlims[0]) & (x_data[:,2] < zlims[1])
        )
        
        return x_data[mask]
    
    
    def generate_children(self):
        """
        Method to generate new children.
        
        """
        
        ### Calc box center coordinates
        xm = (self.xlims[0] + self.xlims[1])/2
        ym = (self.ylims[0] + self.ylims[1])/2
        zm = (self.zlims[0] + self.zlims[1])/2

        ### Parse through bounds of each new child box
        for xlims in [[self.xlims[0], xm], [xm, self.xlims[1]]]:
            for ylims in [[self.ylims[0], ym], [ym, self.ylims[1]]]:
                for zlims in [[self.zlims[0], zm], [zm, self.zlims[1]]]:
                    
                    # Generate new trajectory subset
                    x_points = box.valid_points(self.x_points, xlims, ylims, zlims)
                    
                    # Generate new children
                    self.children.append(box(self.i+1, xlims, ylims, zlims, x_points))
    
    
def count_boxes(root, max_depth=10):
    """
    Box counting algorithm.
    
    Args:
        root (box): Top-level box
        max_depth (int): Max octotree depth
        
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - epsilon: Octree levels.
            - N: Number of boxes at each level.
    """
    
    ### Calculate box dimensions at each level octadically
    epsilon0 = abs(root.xlims[1] - root.xlims[0])
    epsilon = epsilon0 / 2**np.arange(max_depth)
    
    ### Preallocate solution
    N = np.zeros(max_depth)

    ### Iterative BFS for box counting
    queue = [root]
    N = np.zeros(max_depth)
    
    while queue:
        curr_box = queue.pop(0)
        
        # Ignore this box if empty, otherwise, increment solution
        if len(curr_box.x_points) == 0: continue
        N[curr_box.i] += 1
        
        # Check that we don't bottom out before doing more BFS stuff
        if curr_box.i < max_depth - 1:
            
            # Create 8 more nested boxes and append them to the queue
            curr_box.generate_children()
            for child_box in curr_box.children:
                queue.append(child_box)
                
    return epsilon, N