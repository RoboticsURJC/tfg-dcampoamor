import sys
from xmlrpc.server import SimpleXMLRPCServer

# Lista de posiciones
poses = [
    [-1.5, -1.57, -1.5, 0, 0, 0],
    [0, -1.57, 0, 0, 0, 0.1],
    [-1.5, -1.57, -1.4, 0, 0, 1],
]

# Posición especial para indicar que no hay más posiciones
done_pose = [0, 0, 0, 0, 0, 0]

def get_next_pose():
    global poses
    if poses:
        pose = poses.pop(0)
        print(f"Moviendo a la posición: {pose}")
        return pose
    else:
        print("Esperando siguiente posición...")
        return done_pose

server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
print("Listening on port 50000...")

server.register_function(get_next_pose, "get_next_pose")

server.serve_forever()

