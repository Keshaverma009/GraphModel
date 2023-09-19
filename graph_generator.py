import os
import math
import csv
import numpy as np
import pyvista as pv
import cadquery as cq
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

    def convert_stps_to_stls(self, stp_folder_path, stl_folder_path):

        if stp_folder_path.endswith("/") is False:
            stp_folder_path += "/"

        if stl_folder_path.endswith("/") is False:
            stl_folder_path += "/"

        file_names = os.listdir(stp_folder_path)
        file_names.sort()

        for file_name in file_names:
            model_stp = cq.importers.importStep(stp_folder_path + file_name)
            cq.exporters.export(model_stp, stl_folder_path + file_name.split(".")[0] + ".stl")

    def get_intersection_volume(self, model1, model2):

        intersection = model2.boolean_intersection(model1)

        # print(intersection)
        # print(intersection.cell_data)
        # print(intersection.cell_data.active_scalars)

        # pl = pv.Plotter()
        # pl.add_mesh(intersection, color="g")
        # pl.show()

        return intersection.volume

    def get_union_volume(self, model1, model2):
        union_model = model1.boolean_union(model2)
        return union_model.volume

    def classify_contact_relationship_by_volume(self, model1, model2):

        have_contact = False

        for i in range(0, self.displacement_vector_num):

            moved_model1 = model1.copy(deep=True)
            moved_model1.translate(self.displacement_vectors[i, :], inplace=True)

            moved_model2 = model2.copy(deep=True)
            moved_model2.translate(self.displacement_vectors[i, :], inplace=True)

            union_volume1 = self.get_union_volume(moved_model1, model2)
            union_volume2 = self.get_union_volume(moved_model2, model1)

            condition1 = (moved_model1.volume + model2.volume - union_volume1) < self.volume_precision
            condition2 = (moved_model2.volume + model1.volume - union_volume2) < self.volume_precision

            if condition1 is False or condition2 is False:
                have_contact = True
                break

        return have_contact

    def classify_contact_relationship_by_distance(self, model1, model2):

        points1 = model1.points
        points2 = model2.points

        points1_normals = model1.point_normals

        points2_normals = model2.point_normals
        normal_magnitudes = np.sqrt(np.sum(points2_normals * points2_normals, axis=1))
        normal_magnitudes = normal_magnitudes.reshape(points2_normals.shape[0], 1)
        points2_normals = points2_normals / normal_magnitudes

        min_distances = np.zeros((points1.shape[0],))

        indices = list()

        for i in range(0, points1.shape[0]):
            distances = points2 - points1[i, :].reshape(1, 3)
            distances = np.sqrt(np.sum(distances * distances, axis=1))
            min_distances[i] = np.min(distances)
            indices.append([i, np.argmin(distances)])

        min_distance = np.min(min_distances)
        min_distance_index = indices[np.argmin(min_distances)]

        point1_index = min_distance_index[0]
        point2_index = min_distance_index[1]

        normal1 = points1_normals[point1_index, :]
        normal2 = points2_normals[point2_index, :]

        point1 = points1[point1_index, :]
        point2 = points2[point2_index, :]

        vector = point2 - point1

        distance_normal1 = np.fabs(np.sum(vector * normal1))
        distance_normal2 = np.fabs(np.sum(vector * normal2))

        distance_array = np.array([min_distance, distance_normal1, distance_normal2])

        return np.min(distance_array) < self.distance_precision

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

    def classify_contact_relationship(self, model1, model2):
        return self.classify_contact_relationship_by_collision(model1, model2)

    def classify_blocking_relationship(self, model1, model2):

        have_blocking = False

        for i in range(0, model1.points.shape[0]):

            point = model1.points[i, :]

            indices = list()

            indices.append(model2.ray_trace(point, np.array([self.limit_distance, point[1], point[2]]))[1])
            indices.append(model2.ray_trace(point, np.array([-self.limit_distance, point[1], point[2]]))[1])
            indices.append(model2.ray_trace(point, np.array([point[0], self.limit_distance, point[2]]))[1])
            indices.append(model2.ray_trace(point, np.array([point[0], -self.limit_distance, point[2]]))[1])
            indices.append(model2.ray_trace(point, np.array([point[0], point[1], self.limit_distance]))[1])
            indices.append(model2.ray_trace(point, np.array([point[0], point[1], -self.limit_distance]))[1])

            intersection_num = 0

            for j in range(0, 6):
                intersection_num += indices[j].shape[0]

            if intersection_num > 0:
                have_blocking = True
                break

        return have_blocking

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

    def check_corner_cases(self):
        cases = list()

        for i in range(0, self.file_num - 2):

            contact_num = np.sum(self.relationship_matrix[i])

            if contact_num > 1:
                indices = list()

                for j in range(i + 1, self.file_num):
                    if self.relationship_matrix[i, j] == 1:
                        indices.append(j)

                combinations = list()

                for j in range(0, len(indices) - 1):
                    k = j + 1
                    while k < len(indices):
                        combinations.append([indices[j], indices[k]])
                        k += 1

                for combination in combinations:
                    if self.relationship_matrix[combination[0], combination[1]] == 1:
                        cases.append([i, combination[0], combination[1]])

        print(cases)

    def construct_dof_matrix(self, model1, model2, joint_coordinate, part):

        dof_matrix = np.ones((3, 4), dtype=int)

        if part == "model1":
            model1_dup = model1.copy(deep=True)
            model2_dup = model2.copy(deep=True)
        else:
            model1_dup = model2.copy(deep=True)
            model2_dup = model1.copy(deep=True)

        model2_trimesh = trimesh.Trimesh(model2_dup.points, model2_dup.faces.reshape((model2_dup.n_faces, 4))[:, 1:])

        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)

        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        for i in range(1, 4):

            orientation_vector = joint_coordinate[i, :]

            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.translate(orientation_vector * self.distance_precision, inplace=True)

            if self.detect_collision(moved_model1, model2_collision_object) is True:
                dof_matrix[i - 1, 0] = 0

            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.translate(-orientation_vector * self.distance_precision, inplace=True)

            if self.detect_collision(moved_model1, model2_collision_object) is True:
                dof_matrix[i - 1, 1] = 0

            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.rotate_vector(orientation_vector, self.angle_precision,
                                       joint_coordinate[0, :], inplace=True)

            if self.detect_collision(moved_model1, model2_collision_object) is True:
                dof_matrix[i - 1, 2] = 0

            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.rotate_vector(orientation_vector, -self.angle_precision,
                                       joint_coordinate[0, :], inplace=True)

            if self.detect_collision(moved_model1, model2_collision_object) is True:
                dof_matrix[i - 1, 3] = 0

        return dof_matrix

    def construct_dof_matrices(self, model1, model2, joint_coordinates, part):

        joint_coordinates = np.zeros((3, 4, 3))

        joint_num = joint_coordinates.shape[0]

        dof_matrices = np.ones((joint_num, 3, 4), dtype=int)

        for i in range(0, joint_num):
            dof_matrix = self.construct_dof_matrix(model1, model2, joint_coordinates[i, :, :], part)
            dof_matrices[i, :, :] = dof_matrix

        return dof_matrices

    def write_relationships_as_csv_file(self, csv_file_name):

        with open(csv_file_name, "w", newline="") as f:

            writer = csv.writer(f)

            for i in range(0, self.file_num - 1):

                j = i + 1

                while j < self.file_num:

                    if self.relationship_matrix[i, j] == 1:
                        part1_id = self.file_names[i].split(".")[0]
                        part2_id = self.file_names[j].split(".")[1]
                        writer.writerow([part1_id, part2_id, 1])

                    j += 1

    def test(self):
        model1 = pv.Cube((0.0, 0.0, 0.0), 5.0, 5.0, 5.0).triangulate()
        model2 = pv.Cube((5.1, 0.0, 0.0), 5.0, 5.0, 5.0).triangulate()

        # print(self.classify_contact_relationship_by_distance(model1, model2))

        # print((model1 + model2).volume)
        #
        # result = model1.boolean_union(model2)
        #
        # print(result.volume)
        #
        # print(self.classify_contact_relationship_by_distance(model1, model2))
        #
        # pl = pv.Plotter()
        # pl.add_mesh(model1, color="b")
        # pl.add_mesh(model2, color="g")
        # pl.show()

    def demo(self):

        model1 = pv.Sphere(5.0, (0.0, 0.0, 0.0))
        model2 = pv.Sphere(5.0, (10.0, 0.0, 0.0))

        pl = pv.Plotter()

        pl.add_points(model1.points, color="b")
        pl.add_points(model2.points, color="r")

        for i in range(0, self.displacement_vector_num):

            moved_model1 = model1.copy(deep=True)
            moved_model1.translate(self.displacement_vectors[i, :] * 200.0, inplace=True)

            pl.add_points(moved_model1.points, color="g")

        pl.show()

    def demo2(self):

        model1 = pv.Sphere(5.0, (0.0, 0.0, 0.0))
        model2 = pv.Sphere(5.0, (12.0, 0.0, 0.0))

        point1 = model1.points[20, :]
        point2 = model1.points[30, :]
        point3 = model1.points[40, :]

        point11 = np.array([7.5, point1[1], point1[2]])
        point21 = np.array([11.0, point2[1], point2[2]])
        point31 = np.array([7.5, point3[1], point3[2]])

        line1 = np.zeros((2, 3))
        line2 = np.zeros((2, 3))
        line3 = np.zeros((2, 3))

        line1[0, :] = point1
        line1[1, :] = point11

        line2[0, :] = point2
        line2[1, :] = point21

        line3[0, :] = point3
        line3[1, :] = point31

        pl = pv.Plotter()

        pl.add_points(model1.points, color="b")
        pl.add_points(model2.points, color="r")
        pl.add_lines(line1, color="g")
        pl.add_lines(line2, color="g")
        pl.add_lines(line3, color="g")

        pl.show()


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    # files = os.listdir("stl-model")
    # files.sort()
    # 1716806X.stp
    # 1986681X.stp
    graph_generator = GraphGenerator()
    graph_generator.classify_parts_contact_relationships("stl-model")

    # model1 = pv.read("stl-model/" + files[0])
    # model2 = pv.read("stl-model/" + files[1])
    # print(graph_generator.classify_contact_relationship(model1, model2))

    # graph_generator.test()
    # graph_generator.demo()
    # graph_generator.demo2()






