import numpy as np  
import cv2
from guassnewtonsolver import GuassNewtonSolver
from mpmath import *

def finding_contour_points_and_normal_vectors(model_image_segmentation_map,original_image_segmentation_map):
    #exrtracting boundary points from both masks
    contours_model, hier_model = cv2.findContours(model_image_segmentation_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_image, hier_image = cv2.findContours(original_image_segmentation_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    P_H_old = contours_model[0]#storing all boundary points of candidate model
    P_I_old = contours_image[0]#storing all boundary points of image
    
    N_H = []
    N_I = []
    #the below code is wirtten to find the top most point in both masks (since that is the starting point of our sequence)
    top_head_model_index=-1
    top_head_model_y = -1
    top_head_image_index=-1
    top_head_image_y = -1
    
    for i,point_H in enumerate(P_H_old):
        if (pointH[1]>top_head_model_y):
            top_head_model_y = pointH[1]
            top_head_model_index = i
            
            
    for i,point_I in enumerate(P_I_old):
            
        if (pointI[1]>top_head_image_y):
            top_head_image_y = pointI[1]
            top_head_image_index = i;
            
    P_H = []
    P_I = []
    #we rearrange the sequence with topmost head point as the first boundary point for both masks
    P_H.extend(P_H_old[top_head_model_index:])
    P_H.extend(P_H_old[:top_head_model_index])
    
    P_I.extend(P_I_old[top_head_image_index:])
    P_I.extend(P_I_old[:top_head_image_index])
    
    
    #below we calculate the normal vector for each point
    p_h = []
    p_i = []
    
    n_h = []
    n_i = []
    
    for i,point in enumerate(P_H):
        [vx,vy,x,y] = cv.fitLine([P_H[(i-1)%len(P_H)],point,P_H[(i+1)%len(P_H)]], cv.DIST_L2,0,0.01,0.01)
        nx = 1
        ny = -(vx*nx)/ny
        w = sqrt(pow(nx,2)+pow(ny,2))
        nx/=w
        ny/=w
        N_H.append([nx, ny])
        
    for i,point in enumerate(P_I):
        [vx,vy,x,y] = cv.fitLine([P_I[(i-1)%len(P_H)],point,P_I[(i+1)%len(P_H)]], cv.DIST_L2,0,0.01,0.01)
        nx = 1
        ny = -(vx*nx)/ny
        w = sqrt(pow(nx,2)+pow(ny,2))
        nx/=w
        ny/=w
        N_I.append([nx, ny])
        
    #then we uniformally select the amount of points required as mentioned in the paper
    selected_H_points = np.random.uniform(0,len(P_H)-1,size = 200)
    selected_I_points = np.random.uniform(0,len(P_I)-1,size = 2000)
    
    for i in selected_H_points:
        n_h.append(N_H[int(i)])
        p_h.append(P_H[int(i)])
                 
                         
    for i in selected_I_points:
        n_i.append(N_I[int(i)])
        p_i.append(P_I[int(i)])
        
    return p_h, p_i, n_h, n_i


def get_transition_probabilities(p_h, p_i, n_h, n_i):
    
    #transition probability is based on the previous state, its observation, the next state and its observation
    #as mentioned by equation 7 of the paper
    
    transition_prob = {}
    
    for i,point_H_initial in enumerate(p_h):
        for j,point_H_final in enumerate(p_h):
            for k,point_I_initial in enumerate(p_i):
                for l,point_I_final in enumerate(p_i):
            
                    E_e = pow( \
                              sqrt(pow(point_H_initial[0]-point_H_final[0],2)+pow(point_H_initial[1]-point_H_final[1],2)) \
                              - sqrt(pow(point_I_initial[0]-point_I_final[0],2)+pow(point_I_initial[1]-point_I_final[1],2)) \
                                  ,2)
                        
                    transition_prob['P_H['+str(i)+']'+'w/'+'P_I['+str(k)+']'+'-->'+'P_H['+str(j)+']'+'w/'+'P_I['+str(l)+']'] = E_e
                    
    return transition_prob

def get_emission_probabilities(p_h, p_i, n_h, n_i):
    
    
    #emission probability is based on the current state and its observation
    #as mentioned by equation 6 of the paper    

    emission_prob = {}
    lambda_n = 10

    for i, point_H, normal_H in enumerate(zip(p_h,n_h)):
        for k, point_I, normal_I in enumerate(zip(p_i,n_i)):

            E_p = ( pow(point_H[0]-point_I[0],2)+pow(point_H[1]-point_I[1],2) )  \
                + lambda_n * pow(1 - (normal_H[0]*normal_I[0] + normal_H[1]*normal_I[1]),2)
                
            emission_prob['P_H['+str(i)+']'+'w/'+'P_I['+str(k)+']'] = E_p
    

    return emission_prob
                
     
    
def viterbi_algorithm(states, observations, trans_prob, emission_prob):
     #The below program computes the best possible mapping of states to observations based on viterbi decoding
     V = [{}]
     for k, ob in enumerate(observations):
         V[0][k] = {"prob": emission_prob['P_H['+str(0)+']'+'w/'+'P_I['+str(k)+']'], "prev": None}#initalization for state zero
   
     for t in range(1, len(states)):#computing trellis table for all remaining states
         V.append({})
         #for k, ob in enumerate(observations):
         
         prev_obs_selected = 0
         for l, ob_final in enumerate(observations):
            #initailisation for current state
            min_prob = V[t - 1][0]["prob"] + trans_prob['P_H['+str(t-1)+']'+'w/'+'P_I['+str(0)+']'+'-->'+'P_H['+str(t)+']'+'w/'+'P_I['+str(l)+']']+ \
                                          emission_prob['P_H['+str(t)+']'+'w/'+'P_I['+str(l)+']']
            
            for k, ob_initial in enumerate(observations):
                total_prob = V[t - 1][k]["prob"] + trans_prob['P_H['+str(t-1)+']'+'w/'+'P_I['+str(k)+']'+'-->'+'P_H['+str(t)+']'+'w/'+'P_I['+str(l)+']']+ \
                                          emission_prob['P_H['+str(t)+']'+'w/'+'P_I['+str(l)+']']
                if total_prob < min_prob:
                    min_prob = total_prob
                    prev_ob_selected = k
            
            #storing the best scenario of seeing that observation in current states based on equation 5 of the paper
            V[t][l] = {"prob": min_prob, "prev": prev_ob_selected}
 
     #now we compute the optimal sequence of observations for the given states
     optimal_observation_sequence = []
     min_prob = V[-1][0]["prob"]
     best_st = None
 
     for k, observation_dictionary in V[-1].items():
        if observation_dictionary["prob"] < min_prob:
            min_prob = observation_dictionary["prob"]
            best_ob = k
     optimal_observation_sequence.append(observations[best_ob])
     previous = best_ob
 
 
     for t in range(len(V) - 2, -1, -1):
        optimal_observation_sequence.insert(0, observations[V[t + 1][previous]["prev"]])
        previous = V[t + 1][previous]["prev"]
        
    
     return optimal_observation_sequence

def wrapping_function_generator(p_h, p_i_sequence):
    #this function computes wrapping funtion of 2d model points to image points using Thin plate Spline
    tps = cv2.createThinPlateSplineShapeTransformer(regularizationParameter = 1000)
    # regularzation parameter value set based on what was mentioned in the paper
    source_shape = np.array (p_h, np.float32)
    transform_shape = np.array (p_i_sequence, np.float32)
    source_shape = source_shape.reshape (1, -1, 2)
    transform_shape = transform_shape.reshape (1, -1, 2)
    matches = [cv2.DMatch(i,i,0) for i in range(len(p_h))]
    
    tps.estimateTransformation(transform_shape,source_shape,matches) 
    
    #the desired wrapping function is returned to be used later on 
    return tps

def Laplacian_operator(Vertex_list: np.ndarray, Vertex_graph):
    #this function applies the discrete mesh Laplactian operator mentioned in [Desbrun et al. 1999]
    summation = np.zeros((Vertex_list.shape[0],Vertex_list.shape[1]), dtype = np.float32)
    Laplacian_vertices = np.zeros((Vertex_list.shape[0],Vertex_list.shape[1]), dtype = np.float32)
    Magnitude_Laplacian_vertices = np.zeros((Vertex_list.shape[0]), dtype = np.float32)
    
    for x in range(Vertex_list.shape[0]):
        for i in range(len(Vertex_graph[Vertex_list[x]]['Neighbours']))
            diff = np.subtract( Vertex_graph[Vertex_list[x]]['Neighbours'][i] - Vertex_list[x])
            cotangent_diff = (cot(Vertex_graph[Vertex_list[x]]['Angles'][i][0]) + cot(Vertex_graph[Vertex_list[x]]['Angles'][i][1])) * diff
        
            summation[x]  = np.add(summation[x],cotangent_diff)
        Laplacian_vertices[x] = summation[x] / (4*Vertex_graph[Vertex_list[x]]['Total_Area'])
        Magnitude_Laplacian_vertices[x] = sqrt( pow(Laplacian_vertexes[x,0],2) + pow(Laplacian_vertexes[x,1],2) + pow(Laplacian_vertexes[x,2],2))
    #we return the magnitude of laplaction coordinate and the laplaction coordinate for each vertex
    return Magnitude_Laplacian_vertexes, Laplacian_vertexes
        
    
def wrapper_function_3D(Vertex_list: np.ndarray,tps):
    #this function is used to wrap the x,y coordinates of the candidate models vertex points
    wrapped_Vertex_list = np.zeros((Vertex_list.shape[0],Vertex_list.shape[1]), dtype = np.float32)
    Vertex_list_2D = np.zeros((Vertex_list.shape[0],Vertex_list.shape[1]-1), dtype = np.float32)
    
    for x in range(Vertex_list.shape[0]):
            Vertex_list_2D[x,0] = Vertex_list[x,0] 
            Vertex_list_2D[x,1] = Vertex_list[x,1] 
            
    Vertex_list_2D.reshape(1, -1, 2)
    
    ret, wrapped_Vertex_list_2D = tps.applyTransformation(Vertex_list_2D)
    
    wrapped_Vertex_list_2D.reshape(-1,2)
    
    for x in range(Vertex_list.shape[0]):
            wrapped_Vertex_list[x,0] = wrapped_Vertex_list_2D[x,0] 
            wrapped_Vertex_list[x,1] = wrapped_Vertex_list_2D[x,1] 
            wrapped_Vertex_list[x,2] = Vertex_list[x,2]
    
    #we return all the vertex points with their x, y coordinates wrapped by the wrapping function
    return wrapped_Vertex_list

if __name__ == "__main__":
    
    model_image_segmentation_map = cv2.imread('model_segmentation_mask.jpg')#[] #importing binary model mask
    model_image_segmentation_map = cv2.resize(model_image_segmentation_map,(800,800)) # resizing to 800x800
    original_image_segmentation_map = cv2.imread('image_segmentation mask.jpg') #importing binary image mask
    original_image_segmentation_map = cv2.resize(original_image_segmentation_map,(800,800))# resizing to 800x800
    
    Vertex_list = np.array() # assuming all the vertexes corresponding to a paticular candidate model is loaded to Vertex_list in as a numpy array
    """
    Vertex graph is a dictionary of dictionary, we load into it details regarding the positioning of all vertices in the candidate model.
    the outer dictionary has its keys being all the vertices
    for each vertex the inner dictionary stores a list of all its neighbour vertices with key: Neighbours
    it also stores a list in which each element is a list of two angles related to the corresponding element in the Neighbors list 
    (These angles are refered to as alpha_j and beta_j , refer to [Desbrun et al. 1999], which will be used for discrete Laplaciation operation on each vertex)
    it also stores a key called: Total_Area, which stores the total area of all the traingular meshes that particular vertex is a part of 
    """
    Vertex_graph = {{}}# 
    
    """
    Boundary Matching section
    """
    #finding_contour_points_and_normal_vectors return uniformaly sampled 200 mask boundary points of candidate model 
    #and 2000 mask boundary points of image
    p_h, p_i, n_h, n_i = finding_contour_points_and_normal_vectors(model_image_segmentation_map,original_image_segmentation_map) #list of numpy array
    
    #we then calculate Transtion probabilty matrix based on the Edge energy term defined by equation 7 in section 6.2 of the paper
    trans_prob  = get_transition_probabilities(p_h, p_i, n_h, n_i)
    
    #we then calculate Emission probabilty matrix based on the Edge point term defined by equation 6 in section 6.2 of the paper
    emission_prob = get_emission_probabilities(p_h, p_i, n_h, n_i)
    
    #Inorder to minimize Equation 5 mentioned in the paper we use Viterbi decoding algorithm on a HMM model, with Candidate Model points 
    #as states and Image points as observations and achieve the appropriate mapping  (M) of model to image points
    p_i_sequence = viterbi_algorithm(p_h, p_i, trans_prob, emission_prob) 
    #output is corresponding mapping sequence of image points to their 
    #counterpart model points stored in 'p_h'
    
    
    
    """
    Wrapping Function section
    """
    #Using Thin-Plate-Spline method to estimate a wrapping function (W) from Model to Image points based on the mapping function we just generated
    tps = wrapping_function_generator(p_h, p_i_sequence)
    
    """
    Deformation Optimisation Section
    """
    #Now we will deform each vertex in the Candidated model such that Equation 9 mentioned in the paper is minimzed using
    #Ineact Guass Newton Method
    #Guass-Newton Method Implemented in gnsolver.py
    GN_deformation_optimizer = GuassNewtonSolver( tps,
                                                  Vertex_graph
                                                  Laplacian_operator = Laplacian_operator,
                                                  wrapper_function_3D = wrapper_function_3D,
                                                  max_iter = 1000,
                                                  tolerance_difference = 10 ** (-16),
                                                  lambda_s = 1
                                                  )
    # Deformed_Vertex_List stores the new postion of each vertex, corresponding to its counter part in vertex_list
    Deformed_Vertex_list = GN_deformation_optimizer.fit(Vertex_list, Vertex_list)#The first parameter(Vertex_list) acts as an input to our Guass Newton Method
    #While the Second parameter refers to the initail guess of what the deformed vertex position might be, we assuming it to be the original postion
    
    
