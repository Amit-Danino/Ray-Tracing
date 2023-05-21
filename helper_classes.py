import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    return vector - 2 * np.dot(vector, normal) * normal

def calc_intersection(nearest_object, min_distance, ray, epsilon):
    if nearest_object is None:
        return None
    
    intersection = ray.origin + min_distance * ray.direction
    if nearest_object.isSphere:
        thisNormal = normalize(intersection - nearest_object.center)
        intersection += 1e-6 * thisNormal
        nearest_object.normal = thisNormal
    else:       
        intersection += 1e-6 * nearest_object.normal
    
    
    return intersection

## Lights    

class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        return Ray(intersection_point, normalize(self.direction))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection):
        return Ray(intersection,normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        self.position = position
        self.direction = direction
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        return Ray(intersection,normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v =  self.get_light_ray(intersection).direction

        return (self.intensity * np.dot(v, self.direction))/ (self.kc + self.kl*d + self.kq * (d**2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for obj in objects:
            result = obj.intersect(self)
            if result is not None:
                t, objection = result
            else:
                continue
                
            if  t < min_distance:
                min_distance = t
                nearest_object = objection        
        
        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)
        self.isSphere = False

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None


class Rectangle(Object3D):
    """
        A rectangle is defined by a list of vertices as follows:
        a _ _ _ _ _ _ _ _ d
         |               |  
         |               |  
         |_ _ _ _ _ _ _ _|
        b                 c
        This function gets the vertices and creates a rectangle object
    """
    def __init__(self, a, b, c, d):
        """
            ul -> bl -> br -> ur
        """
        self.abcd = [np.asarray(v) for v in [a, b, c, d]]
        self.normal = self.compute_normal()
        self.isSphere = False
        self.a = self.abcd[0]
        self.b = self.abcd[1]
        self.c = self.abcd[2]
        self.d = self.abcd[3]

    def compute_normal(self):
        n = np.cross(self.abcd[1] - self.abcd[0], self.abcd[3] - self.abcd[0])
        return normalize(n)

    # Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        
        plane = Plane(self.normal, self.a)
        result = plane.intersect(ray)
        if result is None:
            return None
        t, _ = result

        # Calculate the intersection point
        intersection_point = calc_intersection(self, t, ray, True) 
        
        # Check if the intersection point is inside the rectangle
        ap, bp, cp, dp = (self.a, self.b, self.c, self.d) - intersection_point

        inside = (
            np.dot(np.cross(ap, bp), self.normal) >= 0 and
            np.dot(np.cross(bp, cp), self.normal) >= 0 and
            np.dot(np.cross(cp, dp), self.normal) >= 0 and
            np.dot(np.cross(dp, ap), self.normal) >= 0
        )
        
        if inside:
            return t, self
        else:
            return None


class Cuboid(Object3D):
    def __init__(self, a, b, c, d, e, f):
        """ 
              g+---------+f
              /|        /|
             / |  E C  / |
           a+--|------+d |
            |Dh+------|B +e
            | /  A    | /
            |/     F  |/
           b+--------+/c
        """
        self.isSphere = False
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        d = np.array(d)
        e = np.array(e)
        f = np.array(f)
        g = a + (f-d)
        h = b + (e-c)

        A = Rectangle(a, b, c, d)
        B = Rectangle(d, c, e, f)
        C = Rectangle(g, h, e, f)
        D = Rectangle(a, b, h, g)
        E = Rectangle(g, a, d, f)
        F = Rectangle(h, b, c, e)
        
        self.face_list = [A,B,C,D,E,F]
                

    def apply_materials_to_faces(self):
        for t in self.face_list:
            t.set_material(self.ambient,self.diffuse,self.specular,self.shininess,self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both
    def intersect(self, ray: Ray):
        t, obj = ray.nearest_intersected_object(self.face_list)
        return obj, t            

class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius
        self.isSphere = True

    def intersect(self, ray: Ray):
        OC = ray.origin - self.center
        D = ray.direction

        a = np.dot(ray.direction, ray.direction)
        b = 2 * (np.dot(D,OC))
        c = np.dot(OC,OC) -(self.radius ** 2)
        
        discriminant = b ** 2 - (4 * a * c)
        if discriminant < 0:
            return None
        
        root = np.sqrt(discriminant)
        result1 = (-b + root) / (2*a)
        result2 = (-b - root) / (2*a)
        
        minresult = [result for result in [result1, result2] if result > 0]


        if len(minresult) > 0:
            return min(minresult), self
        
        return None