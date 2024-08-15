import numpy as np
import matplotlib.pyplot as plt

def system_function(x, y, parameter):
    """Define the system of equations."""
    return np.array([
        x**2 + y**2 - parameter,
        x - y
    ])

def jacobian(x, y):
    """Compute the Jacobian matrix."""
    return np.array([
        [2*x, 2*y],
        [1, -1]
    ])

def continuation_method(initial_point, initial_parameter, num_steps, step_size):
    """Implement the continuation method."""
    x, y = initial_point
    parameter = initial_parameter
    
    boundary_points = [(x, y)]
    parameters = [parameter]
    
    for _ in range(num_steps):
        # Predictor step
        tangent = np.linalg.solve(jacobian(x, y), -np.array([2*x, 2*y]))
        tangent /= np.linalg.norm(tangent)
        
        x_pred = x + step_size * tangent[0]
        y_pred = y + step_size * tangent[1]
        parameter_pred = parameter + step_size
        
        # Corrector step (Newton's method)
        for _ in range(5):  # Max 5 iterations
            F = system_function(x_pred, y_pred, parameter_pred)
            if np.linalg.norm(F) < 1e-6:
                break
            
            J = jacobian(x_pred, y_pred)
            delta = np.linalg.solve(J, -F)
            x_pred += delta[0]
            y_pred += delta[1]
        
        x, y = x_pred, y_pred
        parameter = parameter_pred
        
        boundary_points.append((x, y))
        parameters.append(parameter)
    
    return np.array(boundary_points), np.array(parameters)

# Example usage
initial_point = (1.0, 1.0)
initial_parameter = 2.0
num_steps = 100
step_size = 0.1

boundary, parameters = continuation_method(initial_point, initial_parameter, num_steps, step_size)

# Plotting
plt.figure(figsize=(10, 8))
plt.plot(boundary[:, 0], boundary[:, 1], 'b-', label='Boundary')
plt.scatter(boundary[0, 0], boundary[0, 1], color='red', label='Start point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Boundary Determined by Continuation Method')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()