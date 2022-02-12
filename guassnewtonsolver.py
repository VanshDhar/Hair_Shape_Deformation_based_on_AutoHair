from typing import Callable
import numpy as np
from numpy.linalg import pinv


class GuassNewtonSolver:
    """
    Inexact Gauss-Newton Method Implmentation for vertex deformation.
    Minimize sum(residual) where residual is defined by equation 9 of the paper
    Note since the deformed postion of the vertices are being changed to achieve minimization, they become the parameters/cofficients/weights of 
    function we are trying to minimize and the original vertex position becomes our input for function
    """

    def __init__(self,
                 tps,
                 Vertex_graph,
                 Laplacian_operator: Callable,
                 wrapper_function_3D: Callable,
                 max_iter: int = 1000,
                 tolerance_difference: float = 10 ** (-16),
                 tolerance: float = 10 ** (-9),
                 init_guess: np.ndarray = None,
                 lambda_s = 1
                 ):
        """
        parameters:
        tps: wrapping function
        Vertex graph: holds crucial information about vertices in the candidate model
        Laplacian_operator: a function used to perform laplacian operation on all given vertices in an np array
        wrapper_function_3D: a function used to wrap the x,y coordinate of all the given vertices in an np array based on a wrapping function 
        provided
        max_iter: Maximum number of iterations for optimization.
        tolerance_difference: Terminate iteration if sum(residual) difference between iterations smaller than tolerance.
        tolerance: Terminate iteration if sum(residual) is smaller than tolerance.
        init_guess: Initial guess for coefficients.
        lambda_s: regularization cofficient of our loss function
        """
        self.tps = tps
        self.Vertex_graph = Vertex_graph
        self.Laplacian_operator = Laplacian_operator
        self.wrapper_function_3D = wrapper_function_3D
        self.max_iter = max_iter
        self.tolerance_difference = tolerance_difference
        self.tolerance = tolerance
        self.coefficients = None
        #self.x = None
        self.Vertex_list = None
        #self.y = None
        self.lambda_s = lambda_s
        self.init_guess = None
        if init_guess is not None:
            self.init_guess = init_guess

    def fit(self,
            Vertex_list: np.ndarray,
            init_guess: np.ndarray ) -> np.ndarray:
        """
        Fit coefficients by minimizing sum(residual)
        Vertex_list: input array
        return: Fitted coefficients/ deformed vector position.
        """

        
        self.Vertex_list = Vertex_list
        if init_guess is not None:
            self.init_guess = init_guess

        self.coefficients = self.init_guess
        prev_loss = np.inf
        for k in range(self.max_iter):
            residual = self._calculate_residual(self.coefficients)
            jacobian = self._calculate_jacobian(self.coefficients, step=10 ** (-6))
            self.coefficients = self.coefficients - self._calculate_pseudoinverse(jacobian) @ residual #Guass newton method parameter update
            #its the same as gradient descent but learning rate is fixed and equal to 1/ second derivative(loss) (based on parameters)
            #the total gradients can be calculated using jacobian matrix
            #overall we need to calclate left pseudoinverse of the jacobian matrix
            loss = np.sum(residual)
            if self.tolerance_difference is not None:
                diff = np.abs(prev_loss - loss)
                if diff < self.tolerance_difference:
                    #Since difference between iterations smaller than tolerance we stop our iterations 
                    return self.coefficients
            if loss < self.tolerance:
                #Since loss is smaller than tolerance we stop our iterations
                return self.coefficients
            prev_loss = loss
        
        #return computed cofficients
        return self.coefficients
    

    def _calculate_residual(self, coefficients: np.ndarray) -> np.ndarray:
        #we calculate residual in this function based on the equation 9 of the paper
        wrapped_Vertex_list = self.wrapper_function_3D(self.Vertex_list,self.tps)#(W(v_i) computed)
        Magnitude_Laplacian_input_vertex, Laplacian_input_vertex = self.Laplacian_operator(self.Vertex_list,self.Vertex_graph)
        #(magnitude of delta(v_i) computed), for orignal vertices
        Magnitude_Laplacian_coefficient_vertex, Laplacian_coefficient_vertex = self.Laplacian_operator(coefficients,self.Vertex_graph)
        #magnitude and value of delta(v_i') computed for deformed vertices
        
        #Below computing first part of equation 9
        res1 = np.subtract(Laplacian_coefficient_vertex,wrapped_Vertex_list)
        sum_of_squares = np.zeros((res1.size/3), dtype = np.float32) 
        
        for x in range(res1.shape[0]):
            sum_of_squares[x] = pow(res1[x,0],2) + pow(res1[x,1],2) + pow(res1[x,2],2)
        
        #Below computing the second part of equation 9
        ratio = np.divide(Magnitude_Laplacian_input_vertex, Magnitude_Laplacian_coefficient_vertex)
        
        ratioed_Laplacian_coefficient_vertex = np.zeros((Laplacian_coefficient_vertex.shape[0],Laplacian_coefficient_vertex.shape[1]), dtype = np.float32)
        
        for x in range(Laplacian_coefficient_vertex.shape[0]):
            ratioed_Laplacian_coefficient_vertex[x,0] = Laplacian_coefficient_vertex[x,0] * ratio[x]
            ratioed_Laplacian_coefficient_vertex[x,1] = Laplacian_coefficient_vertex[x,1] * ratio[x]
            ratioed_Laplacian_coefficient_vertex[x,2] = Laplacian_coefficient_vertex[x,2] * ratio[x]
            
        res2 = np.subtract(Laplacian_coefficient_vertex,ratioed_Laplacian_coefficient_vertex)
        
        regularization = np.zeros((res2.size/3), dtype = np.float32)
        
        for x in range(res2.shape[0]):
            regularization[x] = pow(res2[x,0],2) + pow(res2[x,1],2) + pow(res2[x,2],2)
        
        
        #computing the total equation for each vertex
        residual_output = np.add(sum_of_squares, self.lambda_s * regularization)
        #returning the computed value
        return residual_output

    def _calculate_jacobian(self,
                            x0: np.ndarray,
                            step: float = 10 ** (-6)) -> np.ndarray:
        
        #Calculating Jacobian matrix 
        y0 = self._calculate_residual(x0)

        jacobian = []
        for i, parameter in enumerate(x0):
            x = x0.copy()
            x[i] += step
            y = self._calculate_residual(x)
            derivative = (y - y0) / step
            jacobian.append(derivative)
        jacobian = np.array(jacobian).T

        return jacobian

    @staticmethod
    def _calculate_pseudoinverse(x: np.ndarray) -> np.ndarray:
        #Calculates left pseudoinverse of Jacobian matrix required for parameter/cofficient update
        return pinv(x.T @ x) @ x.T