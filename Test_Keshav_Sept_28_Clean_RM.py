import os
import math
import numpy as np
import pyvista as pv
import trimesh
import fcl

class GraphGenerator:

    def __init__(self):

        self.distance_precision = 0.01
        self.angle_precision = 1.0
        self.volume_precision = 0.01
        self.displacement_vector_num = 14

        sqrt_3 = math.sqrt(3)

        self.displacement_vectors = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
                                              [0.0, 1.0, 0.0], [0.0, -1.0, 0.0],
                                              [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
                                              [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                                              [1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [-1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                                              [1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3]])

        self.displacement_vectors *= self.distance_precision

        self.limit_distance = 50000.0

        self.file_folder_path = None
        self.file_names = None
        self.file_num = 0

        self.relationship_matrix = None

    def classify_parts_contact_relationships(self, folder_path):

        if folder_path.endswith("/") is False:
            folder_path += "/"

        file_names = os.listdir(folder_path)
        file_names.sort()

        self.file_folder_path = folder_path
        self.file_names = file_names

        self.file_num = len(self.file_names)

        self.relationship_matrix = np.zeros((self.file_num, self.file_num))

        for i in range(0, self.file_num - 1):

            model1 = pv.read(self.file_folder_path + self.file_names[i])

            j = i + 1

            while j < self.file_num:

                print(f"{i}   {j}   {self.file_names[i]}   {self.file_names[j]}")

                model2 = pv.read(self.file_folder_path + self.file_names[j])

                if self.classify_contact_relationship(model1, model2) is True:
                    self.relationship_matrix[i, j] = 1

                j += 1

        print(self.relationship_matrix)
    
    def classify_contact_relationship(self, model1, model2):
        return self.classify_contact_relationship_by_collision(model1, model2)

    def classify_contact_relationship_by_collision(self, model1, model2):

        model2_trimesh = trimesh.Trimesh(model2.points, model2.faces.reshape((model2.n_faces, 4))[:, 1:])

        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)

        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        have_contact = False

        for i in range(0, self.displacement_vector_num):

            moved_model1 = model1.copy(deep=True)
            moved_model1.translate(self.displacement_vectors[i, :], inplace=True)

            if self.detect_collision(moved_model1, model2_collision_object) is True:
                have_contact = True
                break

        return have_contact

    def detect_collision(self, moved_model1, model2_collision_object):

        moved_model1 = trimesh.Trimesh(moved_model1.points,
                                       moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:])

        moved_model1_bvh = trimesh.collision.mesh_to_BVH(moved_model1)

        moved_model1_collision_object = fcl.CollisionObject(moved_model1_bvh, fcl.Transform())

        collision_request = fcl.CollisionRequest()
        collision_result = fcl.CollisionResult()

        collision_information = fcl.collide(moved_model1_collision_object, model2_collision_object,
                                            collision_request, collision_result)

        return collision_result.is_collision
    
# Example usage:
if __name__ == "__main__":
    generator = GraphGenerator()
    # Calculate and display the relationship matrix
    generator.classify_parts_contact_relationships(folder_path="C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/Sept_18_Test/STL for Matrix")
