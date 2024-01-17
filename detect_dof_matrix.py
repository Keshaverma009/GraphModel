import os
import numpy as np
import pyvista as pv
import trimesh
import fcl
import meshlib.mrmeshpy as mm


translational_distance_num = 6
rotational_angle_num = 6
translational_distances = np.array([1.0, 5.0, 10.0, 20.0, 50.0, 100.0])
rotational_angles = np.array([15.0, 45.0, 75.0, 90.0, 120.0, 150.0])
volume_precision = 0.1


def detect_collision(moved_model1, model2_collision_object):

    moved_model1_trimesh = trimesh.Trimesh(moved_model1.points,
                                           moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:])

    moved_model1_bvh = trimesh.collision.mesh_to_BVH(moved_model1_trimesh)

    moved_model1_collision_object = fcl.CollisionObject(moved_model1_bvh, fcl.Transform())

    collision_request = fcl.CollisionRequest()
    collision_result = fcl.CollisionResult()

    collision_information = fcl.collide(moved_model1_collision_object, model2_collision_object,
                                        collision_request, collision_result)

    return collision_result.is_collision


def compute_joint_coordinate_origin(joint_surface):
    return np.mean(joint_surface.points, axis=0)


def construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup):

    if detect_collision(moved_model1, model2_collision_object) is False:
        print("have no intersection.")
        return 1
    else:
        moved_model1.save("1.stl")
        model2_dup.save("2.stl")
        model1 = mm.loadMesh("1.stl")
        model2 = mm.loadMesh("2.stl")
        intersection_volume = mm.boolean(model1, model2, mm.BooleanOperation.Intersection).mesh.volume()
        if intersection_volume > volume_precision:
            print(f"have intersection. intersection volume: {intersection_volume}.")
            return 0
        else:
            print("have no intersection.")
            return 1


def construct_dof_matrix(model1, model2, joint_coordinate, part):

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

    original_point = joint_coordinate[0, :]

    indicators = ["x", "y", "z"]

    for i in range(1, 4):

        orientation_vector = joint_coordinate[i, :]

        print(f"considering T" + indicators[i - 1] + "+")

        degree_of_freedom = 1

        for j in range(0, translational_distance_num):
            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.translate(orientation_vector * translational_distances[j], inplace=True)
            if construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup) == 0:
                degree_of_freedom = 0
                break

        dof_matrix[i - 1, 0] = degree_of_freedom

        print(f"considering T" + indicators[i - 1] + "-")

        degree_of_freedom = 1

        for j in range(0, translational_distance_num):
            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.translate(-orientation_vector * translational_distances[j], inplace=True)
            if construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup) == 0:
                degree_of_freedom = 0
                break

        dof_matrix[i - 1, 1] = degree_of_freedom

        print(f"considering R" + indicators[i - 1] + "+")

        degree_of_freedom = 1

        for j in range(0, rotational_angle_num):
            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.rotate_vector(orientation_vector, rotational_angles[j], original_point, inplace=True)
            if construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup) == 0:
                degree_of_freedom = 0
                break

        dof_matrix[i - 1, 2] = degree_of_freedom

        print(f"considering R" + indicators[i - 1] + "-")

        degree_of_freedom = 1

        for j in range(0, rotational_angle_num):
            moved_model1 = model1_dup.copy(deep=True)
            moved_model1.rotate_vector(orientation_vector, -rotational_angles[j], original_point, inplace=True)
            if construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup) == 0:
                degree_of_freedom = 0
                break

        dof_matrix[i - 1, 3] = degree_of_freedom

    return dof_matrix


def compute_dof_matrix(model1_path, model2_path, joint_surface_path, part,
                       coordinate_translation=np.zeros((3,)), coordinate_rotation=np.zeros((3,))):

    model1 = pv.read(model1_path)
    model2 = pv.read(model2_path)

    # joint_surface = pv.read(joint_surface_path)
    # joint_origin = self.compute_joint_coordinate_origin(joint_surface)

    joint_origin = np.array([0.0, 0.0, 0.0])
    joint_origin = joint_origin + coordinate_translation

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    if coordinate_rotation[0] != 0.0:
        if coordinate_rotation[0] == 90.0:
            y_axis = np.array([0.0, 0.0, 1.0])
            z_axis = np.array([0.0, -1.0, 0.0])
        else:
            y_axis = np.array([0.0, 0.0, -1.0])
            z_axis = np.array([0.0, 1.0, 0.0])
    elif coordinate_rotation[1] != 0.0:
        if coordinate_rotation[1] == 90.0:
            x_axis = np.array([0.0, 0.0, -1.0])
            z_axis = np.array([1.0, 0.0, 0.0])
        else:
            x_axis = np.array([0.0, 0.0, 1.0])
            z_axis = np.array([-1.0, 0.0, 0.0])
    elif coordinate_rotation[2] != 0.0:
        if coordinate_rotation[2] == 90.0:
            x_axis = np.array([0.0, 1.0, 0.0])
            y_axis = np.array([-1.0, 0.0, 0.0])
        else:
            x_axis = np.array([0.0, -1.0, 0.0])
            y_axis = np.array([1.0, 0.0, 0.0])

    joint_coordinate = np.array([joint_origin, x_axis, y_axis, z_axis])

    return construct_dof_matrix(model1, model2, joint_coordinate, part)


if __name__ == "__main__":

    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    model_paths = ["test_geometries/Part1.stl", "test_geometries/Part2.stl",
                   "test_geometries/Part3.stl", "test_geometries/Part4.stl"]

    print(compute_dof_matrix(model_paths[0], model_paths[2], "test_geometries/Joint1_3.stl", "model2"))
