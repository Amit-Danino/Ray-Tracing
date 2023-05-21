from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            nearest_object, min_distance = ray.nearest_intersected_object(objects)
            intersection = calc_intersection(nearest_object, min_distance, ray, epsilon = True)
            color = get_color(nearest_object, intersection, ambient, lights, ray, objects, 1, max_depth)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)
            

    return image
def get_color(nearest_object, intersection, ambient, lights, ray, objects, current_depth, max_depth):
    if intersection is None:
        return np.zeros(3)
    
    Ka_Ia = ambient * nearest_object.ambient
    
    color = np.float64(Ka_Ia)

    for light in lights:
        light_ray = light.get_light_ray(intersection)
        shadowed = isPixelShadowed(light, light_ray, objects)
        if not shadowed:
            IL_J = light.get_intensity(intersection)
            L_J = light.get_light_ray(intersection).direction
            K_D = nearest_object.diffuse
            left_Part = K_D * IL_J * np.dot(nearest_object.normal, L_J)

            K_S = nearest_object.specular
            L_Jhat = normalize(reflected(L_J, nearest_object.normal))
            V = normalize(ray.origin - intersection)
            shininess = np.floor(nearest_object.shininess / 15.0)
            right_Part = K_S * IL_J * (np.power(np.dot(V, L_Jhat),shininess))
            color += left_Part + right_Part
    
    current_depth += 1
    if current_depth > max_depth:
        return color
    
    reflected_ray = Ray(intersection, reflected(ray.direction, nearest_object.normal))
    new_nearest_object, new_distance = reflected_ray.nearest_intersected_object(objects)
    new_intersection = calc_intersection(new_nearest_object, new_distance, reflected_ray, True)
    
    color += nearest_object.reflection * get_color(new_nearest_object, new_intersection, ambient, lights, reflected_ray, objects, current_depth, max_depth)
    
    return color

def isPixelShadowed(light, ray, objects):
    shadow_obj, shadow_distance = ray.nearest_intersected_object(objects)
    return shadow_obj is not None and shadow_distance <= light.get_distance_from_light(ray.origin)


# Write your own objects and lights
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
