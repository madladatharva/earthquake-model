import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import *
from objloader import OBJ
import math
import csv

def latlon_to_xyz(lat, lon, radius):
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    x = radius * math.cos(lat_r) * math.cos(lon_r)
    y = radius * math.sin(lat_r)
    z = radius * math.cos(lat_r) * math.sin(lon_r)
    return x, y, -z

def load_points_for_year(year):
    actual_pts = []
    pred_pts = []

    with open("cords.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("time", "")
            if len(t) < 4:
                continue

            if t[:4] != str(year):
                continue

            try:
                a_lat = float(row["Actual_Lat"])
                a_lon = float(row["Actual_Lon"])
                p_lat = float(row["Pred_Lat"])
                p_lon = float(row["Pred_Lon"])
            except:
                continue

            ax, ay, az = latlon_to_xyz(a_lat, a_lon, 1.0)  # Increased radius slightly
            px, py, pz = latlon_to_xyz(p_lat, p_lon, 1.0)

            actual_pts.append((ax, ay, az))
            pred_pts.append((px, py, pz))

    return actual_pts, pred_pts

def draw_points(actual, predicted, zoom_factor):
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glDisable(GL_LIGHTING)  # Important: disable lighting for points
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

    # Make points larger and brighter
    base_size = 12.0 / max(1, zoom_factor * 0.5)

    # Draw actual points (BRIGHT RED)
    for x, y, z in actual:
        # Glow effect
        glColor4f(1, 0.2, 0.2, 0.6)  # Brighter red
        glPointSize(base_size * 2.0)
        glBegin(GL_POINTS)
        glVertex3f(x, y, z)
        glEnd()
        
        # Main point - very bright
        glColor3f(1, 0, 0)
        glPointSize(base_size)
        glBegin(GL_POINTS)
        glVertex3f(x, y, z)
        glEnd()

    # Draw predicted points (BRIGHT BLUE)
    for x, y, z in predicted:
        # Glow effect
        glColor4f(0.2, 1, 0.2, 0.6)  # Brighter green
        glPointSize(base_size * 2.0)
        glBegin(GL_POINTS)
        glVertex3f(x, y, z)
        glEnd()
        
        # Main point - very bright
        glColor3f(0, 1, 0)
        glPointSize(base_size)
        glBegin(GL_POINTS)
        glVertex3f(x, y, z)
        glEnd()

    glPopAttrib()

# Initialize
pygame.init()
display = (800, 600)
screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("3D Globe with Year Slider")

# Initialize OpenGL
gluPerspective(45, display[0]/display[1], 0.1, 50.0)
glEnable(GL_DEPTH_TEST)
glEnable(GL_POINT_SMOOTH)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Remove or comment out the line that makes the globe semi-transparent
# glColor4f(1, 1, 1, 0.7)  # Remove or comment this line

# Load 3D model
model = OBJ("globe.obj", swapyz=False)

# Camera and rotation variables
zoom = -5  # Start closer
glTranslatef(0, 0, zoom)

# Year slider variables - Adjusted to match your data
min_year = 2006  # Changed from 1980
max_year = 2011
current_year = 2006  # Start at first year with data
slider_x = 100
slider_y = 500
slider_width = 600
slider_height = 20
handle_radius = 10
dragging = False

# Mouse and rotation variables
last_x = 0
last_y = 0
mouse_held = False
rotate = True
angle = 0
angle_y = 0
running = True

# Load initial points
actual_points, predicted_points = load_points_for_year(current_year)

# For 2D text rendering
pygame.font.init()
font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

def draw_text_overlay():
    """Draw 2D text overlay"""
    # Switch to 2D orthographic projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, display[0], display[1], 0)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    # Disable depth test for 2D drawing
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    
    # Draw semi-transparent background for text area
    glColor4f(0.0, 0.0, 0.0, 0.5)
    glBegin(GL_QUADS)
    glVertex2f(0, 0)
    glVertex2f(display[0], 0)
    glVertex2f(display[0], 60)
    glVertex2f(0, 60)
    glEnd()
    
    # Restore OpenGL state
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # Check if clicking on slider handle
                handle_x = slider_x + ((current_year - min_year) / (max_year - min_year)) * slider_width
                handle_y = slider_y + slider_height // 2
                
                distance = ((mouse_x - handle_x) ** 2 + (mouse_y - handle_y) ** 2) ** 0.5
                
                if distance <= handle_radius:
                    dragging = True
                else:
                    # If not on slider, rotate globe
                    mouse_held = True
                    last_x, last_y = mouse_x, mouse_y
                    
            elif event.button == 4:  # Wheel up - zoom in
                zoom += 0.5
                glTranslatef(0, 0, 0.5)
            elif event.button == 5:  # Wheel down - zoom out
                zoom -= 0.5
                glTranslatef(0, 0, -0.5)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                mouse_held = False
                dragging = False
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                rotate = not rotate
            elif event.key == pygame.K_LEFT:
                current_year = max(min_year, current_year - 1)
                actual_points, predicted_points = load_points_for_year(current_year)  # Fixed this line
            elif event.key == pygame.K_RIGHT:
                current_year = min(max_year, current_year + 1)
                actual_points, predicted_points = load_points_for_year(current_year)  # Fixed this line
            elif event.key == pygame.K_r:  # Reset view
                angle = 0
                angle_y = 0
                zoom = -5
                glLoadIdentity()
                gluPerspective(45, display[0]/display[1], 0.1, 50.0)
                glTranslatef(0, 0, zoom)
    
    # Update slider if dragging
    if dragging:
        mouse_x = pygame.mouse.get_pos()[0]
        mouse_x = max(slider_x, min(mouse_x, slider_x + slider_width))
        
        percent = (mouse_x - slider_x) / slider_width
        new_year = int(min_year + percent * (max_year - min_year))
        
        if new_year != current_year:
            current_year = new_year
            actual_points, predicted_points = load_points_for_year(current_year)  # Fixed this line
    
    # Handle globe rotation
    if mouse_held and not dragging:
        x, y = pygame.mouse.get_pos()
        diff = x - last_x
        diff_y = y - last_y
        angle_y += diff_y * 0.5
        angle += diff * 0.5
        last_x = x
        last_y = y
    
    # Auto-rotate
    if rotate and not mouse_held and not dragging:
        angle += 0.05
    
    # Clear screen
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Draw 3D scene
    glPushMatrix()
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle, 0, 1, 0)
    
    # Draw globe model - simple solid color
    glDisable(GL_LIGHTING)  # Disable lighting for simple globe
    glColor3f(0.2, 0.3, 0.8)  # Simple blue color
    model.render()
    
    # Draw points on top
    draw_points(actual_points, predicted_points, abs(zoom) + 1)
    glPopMatrix()
    
    # === DRAW SLIDER IN 2D OVERLAY ===
    # Switch to 2D orthographic projection
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(0, display[0], display[1], 0)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Draw semi-transparent background for slider area
    glColor4f(0.0, 0.0, 0.0, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(0, slider_y - 40)
    glVertex2f(display[0], slider_y - 40)
    glVertex2f(display[0], slider_y + 60)
    glVertex2f(0, slider_y + 60)
    glEnd()
    
    # Draw slider track
    glColor3f(0.4, 0.4, 0.4)
    glBegin(GL_QUADS)
    glVertex2f(slider_x, slider_y)
    glVertex2f(slider_x + slider_width, slider_y)
    glVertex2f(slider_x + slider_width, slider_y + slider_height)
    glVertex2f(slider_x, slider_y + slider_height)
    glEnd()
    
    # Draw slider border
    glColor3f(0.8, 0.8, 0.8)
    glLineWidth(2.0)
    glBegin(GL_LINE_LOOP)
    glVertex2f(slider_x, slider_y)
    glVertex2f(slider_x + slider_width, slider_y)
    glVertex2f(slider_x + slider_width, slider_y + slider_height)
    glVertex2f(slider_x, slider_y + slider_height)
    glEnd()
    
    # Draw slider handle
    handle_x = slider_x + ((current_year - min_year) / (max_year - min_year)) * slider_width
    handle_y = slider_y + slider_height // 2
    
    glColor3f(0.0, 0.6, 1.0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(handle_x, handle_y)
    for i in range(360):
        angle_rad = math.radians(i)
        x = handle_x + math.cos(angle_rad) * handle_radius
        y = handle_y + math.sin(angle_rad) * handle_radius
        glVertex2f(x, y)
    glEnd()
    
    # Draw handle highlight
    glColor3f(0.8, 0.9, 1.0)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(handle_x, handle_y)
    for i in range(360):
        angle_rad = math.radians(i)
        x = handle_x + math.cos(angle_rad) * (handle_radius - 3)
        y = handle_y + math.sin(angle_rad) * (handle_radius - 3)
        glVertex2f(x, y)
    glEnd()
    
    # Restore OpenGL state
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    # Create a surface for text
    text_surface = pygame.Surface(display, pygame.SRCALPHA)
    text_surface.fill((0, 0, 0, 0))
    
    # Year text
    year_text = font.render(f"Year: {current_year}", True, (255, 255, 255))
    text_surface.blit(year_text, (display[0] // 2 - 50, 20))
    
    # Points count
    count_text = font.render(f"Actual: {len(actual_points)}  Predicted: {len(predicted_points)}", 
                            True, (255, 255, 255))
    text_surface.blit(count_text, (display[0] // 2 - 100, 50))
    
    # Instructions
    instr_text = small_font.render("W: Toggle auto-rotate | R: Reset view | Arrow Keys: Change year", 
                                  True, (200, 200, 200))
    text_surface.blit(instr_text, (10, display[1] - 30))
    
    # Legend
    pygame.draw.circle(text_surface, (255, 0, 0), (20, 30), 5)
    legend_actual = small_font.render("Actual", True, (255, 255, 255))
    text_surface.blit(legend_actual, (30, 25))
    
    pygame.draw.circle(text_surface, (0, 255, 0), (100, 30), 5)
    legend_pred = small_font.render("Predicted", True, (255, 255, 255))
    text_surface.blit(legend_pred, (110, 25))
    
    # Switch to 2D mode to draw the text surface
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, display[0], display[1], 0, -1, 1)
    
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Draw the text surface as a texture
    texture_data = pygame.image.tostring(text_surface, "RGBA", True)
    
    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, display[0], display[1], 
                 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    
    glEnable(GL_TEXTURE_2D)
    glBegin(GL_QUADS)
    glTexCoord2f(0, 1); glVertex2f(0, 0)
    glTexCoord2f(1, 1); glVertex2f(display[0], 0)
    glTexCoord2f(1, 0); glVertex2f(display[0], display[1])
    glTexCoord2f(0, 0); glVertex2f(0, display[1])
    glEnd()
    glDisable(GL_TEXTURE_2D)
    
    # Clean up
    glDeleteTextures([tex_id])
    
    # Restore OpenGL state
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    
    # Single flip at the end
    pygame.display.flip()
    pygame.time.wait(10)

# Cleanup
model.free()
pygame.quit()
