"""Offscreen VTK renderer for checking the model."""
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray


def mesh_of(shape, tol=0.15, ang=0.3):
    vs, fs = shape.tessellate(tol, ang)
    V = np.array([(v.x, v.y, v.z) for v in vs], dtype=np.float64)
    F = np.array(fs, dtype=np.int64).reshape(-1, 3)
    return V, F


def polydata(V, F):
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(V, deep=True))
    cells = np.hstack([np.full((len(F), 1), 3, np.int64), F]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(F), numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    return pd


def render(parts, path, view=(1, -1, 0.7), up=(0, 0, 1), size=(1400, 1000),
           clip=None, zoom=1.0, focal=None, meshes=None, ortho=False, title=None):
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.96, 0.96, 0.97)
    items = meshes if meshes is not None else [(mesh_of(p.world_shape()), p.color) for p in parts]
    for (V, F), col in items:
        pd = polydata(V, F)
        src = pd
        if clip is not None:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*clip[0])
            plane.SetNormal(*clip[1])
            cl = vtk.vtkClipPolyData()
            cl.SetInputData(pd)
            cl.SetClipFunction(plane)
            cl.Update()
            src = cl.GetOutput()
        norms = vtk.vtkPolyDataNormals()
        norms.SetInputData(src)
        norms.SetFeatureAngle(35)
        m = vtk.vtkPolyDataMapper()
        m.SetInputConnection(norms.GetOutputPort())
        a = vtk.vtkActor()
        a.SetMapper(m)
        a.GetProperty().SetColor(*col)
        a.GetProperty().SetSpecular(0.25)
        a.GetProperty().SetSpecularPower(20)
        if clip is not None:
            a.GetProperty().BackfaceCullingOff()
        ren.AddActor(a)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetSize(*size)
    win.AddRenderer(ren)
    cam = ren.GetActiveCamera()
    ren.ResetCamera()
    fp = np.array(focal if focal is not None else cam.GetFocalPoint())
    dist = cam.GetDistance()
    d = np.array(view, float)
    d /= np.linalg.norm(d)
    cam.SetFocalPoint(*fp)
    cam.SetPosition(*(fp + d * dist))
    cam.SetViewUp(*up)
    if ortho:
        cam.ParallelProjectionOn()
    ren.ResetCamera()
    if focal is not None:
        cam.SetFocalPoint(*fp)
        cam.SetPosition(*(fp + d * cam.GetDistance()))
    cam.Zoom(zoom)
    ren.ResetCameraClippingRange()
    if title:
        t = vtk.vtkTextActor()
        t.SetInput(title)
        t.GetTextProperty().SetFontSize(22)
        t.GetTextProperty().SetColor(0.1, 0.1, 0.1)
        t.SetPosition(12, 12)
        ren.AddActor2D(t)
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.Update()
    wr = vtk.vtkPNGWriter()
    wr.SetFileName(path)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()
    return path


_mesh_cache = {}


def local_mesh(part, tol=0.2, ang=0.35):
    k = id(part.shape)
    if k not in _mesh_cache:
        _mesh_cache[k] = (mesh_of(part.local_shape(), tol, ang), part.shape)
    return _mesh_cache[k][0]


def world_meshes(parts, tol=0.2):
    out = []
    for p in parts:
        V, F = local_mesh(p, tol)
        Vh = np.c_[V, np.ones(len(V))] @ p.M.T
        out.append(((Vh[:, :3], F), p.color))
    return out
