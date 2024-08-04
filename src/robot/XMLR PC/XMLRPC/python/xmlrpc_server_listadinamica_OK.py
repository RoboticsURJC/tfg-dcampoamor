import sys
from xmlrpc.server import SimpleXMLRPCServer

poses = [
    [-1.5, -1.57, -1.5, 0, 0, 0], 
    [0, -1.57, 0, 0, 0, 0.1], 
    [-1.5, -1.57, -1.4, 0, 0, 1],  
]

current_index = 0

def get_next_pose():
    global current_index
    if current_index < len(poses):
        pose = poses[current_index]
        current_index += 1
        print(f"Sending pose: {pose}")
        return pose
    else:
        print("No more poses left")
        return "No more poses"

server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
print("Listening on port 50000...")

server.register_function(get_next_pose, "get_next_pose")

server.serve_forever()

