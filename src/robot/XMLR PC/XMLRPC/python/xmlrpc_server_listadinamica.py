import sys
from xmlrpc.server import SimpleXMLRPCServer

poses = [
    [-1.5, -1.57, -1.5, 0, 0, 0], 
    [0, -1.57, 0, 0, 0, 0.1], 
    [-1.5, -1.57, -1.4, 0, 0, 1],  
]

def get_next_pose():
    global poses
    if poses:
        # Get the next pose
        pose = poses.pop(0)
        print(f"Sending pose: {pose}")
        return pose
    else:
        # Return a string "None" when no more poses are left
        print("No more poses left")
        return "None"

server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
print("Listening on port 50000...")

server.register_function(get_next_pose, "get_next_pose")

server.serve_forever()


