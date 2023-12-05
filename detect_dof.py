import numpy as np
import trimesh
import fcl
import meshlib.mrmeshpy as mm
import pyvista as pv


translation_distance = 1.0
rotation_angle = 20.0
volume_precision = 0.01


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

    # joint_coordinate = np.zeros((4, 3))

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
        moved_model1 = model1_dup.copy(deep=True)
        moved_model1.translate(orientation_vector * translation_distance, inplace=True)
        dof_matrix[i - 1, 0] = construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup)

        print(f"considering T" + indicators[i - 1] + "-")
        moved_model1 = model1_dup.copy(deep=True)
        moved_model1.translate(-orientation_vector * translation_distance, inplace=True)
        dof_matrix[i - 1, 1] = construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup)

        print(f"considering R" + indicators[i - 1] + "+")
        moved_model1 = model1_dup.copy(deep=True)
        moved_model1.rotate_vector(orientation_vector, rotation_angle, original_point, inplace=True)
        dof_matrix[i - 1, 2] = construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup)

        print(f"considering R" + indicators[i - 1] + "-")
        moved_model1 = model1_dup.copy(deep=True)
        moved_model1.rotate_vector(orientation_vector, -rotation_angle, original_point, inplace=True)
        dof_matrix[i - 1, 3] = construct_dof_matrix_element(moved_model1, model2_collision_object, model2_dup)

    return dof_matrix


if __name__ == "__main__":
    model1 = pv.Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.5, 5.0).triangulate()
    model2 = pv.Cylinder((5.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.5, 5.0).triangulate()

    joint_coordinate = np.array([[2.5, 0, 0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0],
                                 [0.0, 0.0, 1.0]])

    print(construct_dof_matrix(model1, model2, joint_coordinate, "model2"))
