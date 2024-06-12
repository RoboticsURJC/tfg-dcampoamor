import sys
import urllib
is_py2 = sys.version[0] == '2'
if is_py2:
    from SimpleXMLRPCServer import SimpleXMLRPCServer
else:
    from xmlrpc.server import SimpleXMLRPCServer

# Lista de poses
poses = [
    [-0.18, -0.61, 0.23, 0, 3.12, 0.04],
    [0.15, 0.45, -0.12, 0.1, 2.1, 0.3],
    [0.3, -0.4, 0.5, 0.2, 1.0, -0.1]
]

# Índice de la pose actual
current_pose_index = 0

def get_next_pose(p):
    global current_pose_index
    assert type(p) is dict
    pose = urllib.poseToList(p)
    print("Received pose: " + str(pose))

    # Obtener la siguiente pose en la lista
    next_pose = poses[current_pose_index]
    print("Sending next pose: " + str(next_pose))
    current_pose_index = (current_pose_index + 1) % len(poses)

    return urllib.listToPose(next_pose)

server = SimpleXMLRPCServer(("", 50000), allow_none=True)
server.RequestHandlerClass.protocol_version = "HTTP/1.1"
print("Listening on port 50000...")

server.register_function(get_next_pose, "get_next_pose")

server.serve_forever()
