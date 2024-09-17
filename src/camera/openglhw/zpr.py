import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from pyrr import Matrix44, Vector3

# Shader sources
vertex_shader_source = """
# version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
"""

fragment_shader_source = """
# version 330 core
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0, 0.5, 0.2, 1.0);
}
"""

# Vertex data
vertices = np.array([
    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,

    -0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5, -0.5,  0.5,

    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5, -0.5,
    -0.5, -0.5,  0.5,
    -0.5,  0.5,  0.5,

     0.5,  0.5,  0.5,
     0.5,  0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
     0.5,  0.5,  0.5,

    -0.5, -0.5, -0.5,
     0.5, -0.5, -0.5,
     0.5, -0.5,  0.5,
     0.5, -0.5,  0.5,
    -0.5, -0.5,  0.5,
    -0.5, -0.5, -0.5,

    -0.5,  0.5, -0.5,
     0.5,  0.5, -0.5,
     0.5,  0.5,  0.5,
     0.5,  0.5,  0.5,
    -0.5,  0.5,  0.5,
    -0.5,  0.5, -0.5,
], dtype=np.float32)

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

window = glfw.create_window(800, 600, "OpenGL Camera", None, None)

if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)

# Compile and link shaders
shader = compileProgram(
    compileShader(vertex_shader_source, GL_VERTEX_SHADER),
    compileShader(fragment_shader_source, GL_FRAGMENT_SHADER)
)

# Create VAO and VBO
VAO = glGenVertexArrays(1)
VBO = glGenBuffers(1)

glBindVertexArray(VAO)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Set vertex attribute pointers
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * vertices.itemsize, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

# Unbind VBO and VAO
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

# Define the camera
camera_pos = Vector3([0.0, 0.0, 3.0])
camera_front = Vector3([0.0, 0.0, -1.0])
camera_up = Vector3([0.0, 1.0, 0.0])

# Set up projection matrix
projection = pyrr.matrix44.create_perspective_projection(45.0, 800/600, 0.1, 100.0)

# Set initial movement and zoom speed
camera_speed = 0.05
zoom_speed = 1.0

def key_input_callback(window, key, scancode, action, mode):
    global camera_pos, camera_front, camera_up, zoom_speed
    
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    
    # Movimiento de la cámara
    if key == glfw.KEY_W and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_pos += camera_speed * camera_front
    if key == glfw.KEY_S and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_pos -= camera_speed * camera_front
    if key == glfw.KEY_A and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_pos -= pyrr.vector.normalize(pyrr.vector.cross(camera_front, camera_up)) * camera_speed
    if key == glfw.KEY_D and (action == glfw.PRESS or action == glfw.REPEAT):
        camera_pos += pyrr.vector.normalize(pyrr.vector.cross(camera_front, camera_up)) * camera_speed
    
    # Zoom de la cámara
    if key == glfw.KEY_UP and action == glfw.PRESS:
        zoom_speed += 0.1
    if key == glfw.KEY_DOWN and action == glfw.PRESS:
        zoom_speed -= 0.1
    
glfw.set_key_callback(window, key_input_callback)

# Main loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    
    # Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # View matrix
    view = pyrr.matrix44.create_look_at(camera_pos, camera_pos + camera_front, camera_up)
    
    # Adjust projection matrix for zoom
    projection_zoomed = pyrr.matrix44.create_perspective_projection(45.0 * zoom_speed, 800/600, 0.1, 100.0)
    
    glUseProgram(shader)
    
    # Set the model, view, and projection matrices in the vertex shader
    model_loc = glGetUniformLocation(shader, "model")
    view_loc = glGetUniformLocation(shader, "view")
    proj_loc = glGetUniformLocation(shader, "projection")

    glUniformMatrix4fv(model_loc, 1, GL_FALSE, pyrr.matrix44.create_identity())
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)
    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection_zoomed)
    
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, len(vertices) // 3)
    glBindVertexArray(0)
    
    glfw.swap_buffers(window)

glfw.terminate()

