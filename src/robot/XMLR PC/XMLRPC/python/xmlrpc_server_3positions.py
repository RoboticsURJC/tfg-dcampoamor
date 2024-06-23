import sys
from xmlrpc.server import SimpleXMLRPCServer

# Define three poses within a compact region for UR3e (in meters)
# Keeping X and Y constant, varying Z and rotation around Z (yaw)
poses = [
    [-0.18, -0.61, 0.23, 0, 3.12, 0.04],  
    [0.15, 0.45, -0.12, 0.1, 2.1, 0.3],  
    [0.3, -0.4, 0.5, 0.2, 1.0, -0.1]   
]
current_index = 0

def get_next_pose():
    global current_index
    pose = poses[current_index]
    print(f"Sending pose: {pose}")
    
    # Update the index to the next pose, wrap around if needed
    current_index = (current_index + 1) % len(poses)
    
    return pose

server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
print("Listening on port 50000...")

server.register_function(get_next_pose, "get_next_pose")

server.serve_forever()
