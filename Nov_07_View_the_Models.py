import vtk
import os

# Directory containing STL files
directory = 'C:/Users/Keshav Verma/Desktop/Work/HIWI GAPP/Graph_Model_Project/NOV_07/test/'

# Create a renderer
ren = vtk.vtkRenderer()

# Create a render window
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)

# Create an interactor
iren = vtk.vtkRenderWindowInteractor()
style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetRenderWindow(renWin)
iren.SetInteractorStyle(style)

# Iterate over STL files in the directory and add them to the renderer
for file in os.listdir(directory):
    if file.endswith('.STL'):
        file_path = os.path.join(directory, file)
        # Read STL file
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_path)

        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(reader.GetOutputPort())

        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add actor to renderer
        ren.AddActor(actor)

# Initialize the interactor and start rendering
iren.Initialize()
renWin.Render()
iren.Start()
