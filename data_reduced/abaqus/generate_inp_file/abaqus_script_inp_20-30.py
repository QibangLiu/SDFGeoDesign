
# %%
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

import os
import numpy as np
import json


# %%
def generate_inp(working_dir):
    os.chdir(working_dir)
    # ###
    sample_file = "sample.json"
    # working_dir = args.working_dir
    solver = 'implicit'
    strain = -0.2
    coarseness = 0.04
    #
    with open(sample_file, 'r') as file:
        geo_data = json.load(file)

    vertices = np.array(geo_data['vertices'])
    inner_loops = geo_data['inner_loops']
    out_loop = geo_data['out_loop']

    x_min, x_max = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    y_min, y_max = np.min(vertices[:, 1]), np.max(vertices[:, 1])
    width, height = x_max-x_min, y_max-y_min
    # ###
    strain = strain*(1+1/50)  # 1/50 is the number of strain steps
    model_name = "MyModel"
    sketch_name = "Sketch-1"
    UCpart_name = "UC_part"
    topline_part_name = "Topline_part"
    bottomline_part_name = "Bottomline_part"
    material_name = "Material-1"
    section_name = "Section-1"

    if model_name in mdb.models:
        del mdb.models[model_name]
        print(f"Model '{model_name}' has been deleted.")

    mdb.Model(name=model_name)
    myModel = mdb.models[model_name]

    mySketch = myModel.ConstrainedSketch(name=sketch_name, sheetSize=2.0)

    for i in range(len(out_loop)-1):
        mySketch.Line(point1=vertices[out_loop[i], :2],
                      point2=vertices[out_loop[i+1], :2])

    for loop in inner_loops:
        for i in range(len(loop)-1):
            mySketch.Line(point1=vertices[loop[i], :2],
                          point2=vertices[loop[i+1], :2])

    # ###
    # Create a unit cell part from the sketch
    UCPart = myModel.Part(
        name=UCpart_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    UCPart.BaseShell(sketch=mySketch)
    del myModel.sketches[sketch_name]

    # creat top and bottom lines for comression
    mySketch = myModel.ConstrainedSketch(name=sketch_name, sheetSize=2*width)
    mySketch.Line(point1=(x_min-width*0.5, y_max),
                  point2=(x_max+width*0.5, y_max))
    myModel.Part(dimensionality=TWO_D_PLANAR,
                 name=topline_part_name, type=ANALYTIC_RIGID_SURFACE)
    myModel.parts[topline_part_name].AnalyticRigidSurf2DPlanar(sketch=mySketch)
    del myModel.sketches[sketch_name]
    myModel.parts[topline_part_name].ReferencePoint(
        point=(x_min+width*0.5, y_max, 0))

    mySketch = myModel.ConstrainedSketch(name=sketch_name, sheetSize=2*width)
    mySketch.Line(point1=(x_min-width*0.5, y_min),
                  point2=(x_max+width*0.5, y_min))
    myModel.Part(dimensionality=TWO_D_PLANAR,
                 name=bottomline_part_name, type=ANALYTIC_RIGID_SURFACE)
    myModel.parts[bottomline_part_name].AnalyticRigidSurf2DPlanar(
        sketch=mySketch)
    del myModel.sketches[sketch_name]
    myModel.parts[bottomline_part_name].ReferencePoint(
        point=(x_min+width*0.5, y_min, 0))
    # ### properties
    # Create a material
    myModel.Material(name=material_name)
    E = 2.306e3
    nu = 0.35
    density = 1e-8
    myModel.materials[material_name].Elastic(table=((E, nu), ))

    myModel.materials[material_name].Density(table=((density, ), ))
    myModel.materials[material_name].Damping(beta=0.005)

    myModel.materials[material_name].Plastic(table=((40.62, 0.0),
                                                    (45.24, 0.001133),
                                                    (52.62, 0.004183),
                                                    (58.00, 0.0080645),
                                                    (61.87, 0.012557),
                                                    (65.81, 0.020035),
                                                    (69.19, 0.030689),
                                                    (71.06, 0.038873),
                                                    (72.61, 0.047114),
                                                    (73.54, 0.052610),
                                                    (74.82, 0.06083),
                                                    (76.74, 0.074477),
                                                    (78.46, 0.08799),
                                                    (81.58, 0.11457),
                                                    (83.00, 0.1276)
                                                    ))

    # create a section
    myModel.HomogeneousSolidSection(
        material=material_name, name=section_name, thickness=None)
    # assign the section to the part
    all_faces_set_name = "UC_faces_Set"
    # Apply to all faces of the part
    UC_faces_Set = UCPart.Set(faces=UCPart.faces[:], name=all_faces_set_name)
    UCPart.SectionAssignment(region=UC_faces_Set, sectionName=section_name)

    # ###  Mesh
    # set element type

    if solver == 'static':
        # set element to plain strain
        UCPart.setElementType(elemTypes=(ElemType(
            elemCode=CPE4, elemLibrary=STANDARD, secondOrderAccuracy=OFF,
            hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
            elemCode=CPE3, elemLibrary=STANDARD)), regions=UC_faces_Set)
    else:
        # set element to plain strain
        UCPart.setElementType(elemTypes=(ElemType(
            elemCode=CPE4, elemLibrary=STANDARD, secondOrderAccuracy=OFF,
            hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
            elemCode=CPE3, elemLibrary=STANDARD)), regions=UC_faces_Set)

    myModel.parts[UCpart_name].seedPart(deviationFactor=0.1,
                                        minSizeFactor=0.1, size=coarseness)
    myModel.parts[UCpart_name].generateMesh()

    # ###
    # Create an assembly
    uc_assem_name = "uc_assem"
    topline_assem_name = "topline_assem"
    bottomline_assem_name = "bottomline_assem"
    myModel.rootAssembly.DatumCsysByDefault(CARTESIAN)
    myModel.rootAssembly.Instance(
        name=uc_assem_name, part=myModel.parts[UCpart_name], dependent=ON)
    myModel.rootAssembly.Instance(
        name=topline_assem_name, part=myModel.parts[topline_part_name], dependent=ON)
    myModel.rootAssembly.Instance(
        name=bottomline_assem_name, part=myModel.parts[bottomline_part_name], dependent=ON)
    root_assem = myModel.rootAssembly
    uc_assem_inst = root_assem.instances[uc_assem_name]

    # find bcs
    # Iterate over all edges and select edges with x-coordinate close to 0
    leftEdges, rightEdges, bottomEdges, topEdges = [], [], [], []
    for edge in uc_assem_inst.edges:
        id1, idx = edge.getVertices()  # Get one vertex (or you can choose midpoint)
        p1, p2 = uc_assem_inst.vertices[id1].pointOn[0], uc_assem_inst.vertices[idx].pointOn[0]
        if np.isclose(p1[0], x_min) and np.isclose(p2[0], x_min):
            leftEdges.append(edge)
        elif np.isclose(p1[0], x_max) and np.isclose(p2[0], x_max):
            rightEdges.append(edge)
        elif np.isclose(p1[1], y_min) and np.isclose(p2[1], y_min):
            bottomEdges.append(edge)
        elif np.isclose(p1[1], y_max) and np.isclose(p2[1], y_max):
            topEdges.append(edge)

    allNodes = uc_assem_inst.nodes
    root_assem.Set(edges=EdgeArray(leftEdges), name="LeftEdgeSet")
    root_assem.Set(edges=EdgeArray(rightEdges), name="RightEdgeSet")
    root_assem.Set(edges=EdgeArray(bottomEdges), name="BottomEdgeSet")
    root_assem.Set(edges=EdgeArray(topEdges), name="TopEdgeSet")
    # ### Step
    # Define a static general step for plasticity

    if solver == 'explicit':
        step_name = "uc_explicit_step"
        myModel.ExplicitDynamicsStep(improvedDtMethod=ON, name=step_name,
                                     previous='Initial')
    elif solver == 'implicit':
        step_name = "uc_implicit_step"
        myModel.ImplicitDynamicsStep(name=step_name,
                                     previous='Initial',
                                     timePeriod=1.0,
                                     nlgeom=ON,
                                     initialInc=1e-3,
                                     minInc=1e-8,
                                     maxNumInc=400,
                                     alpha=DEFAULT,
                                     amplitude=RAMP,
                                     application=MODERATE_DISSIPATION,
                                     initialConditions=OFF)
    elif solver == 'static':
        step_name = "uc_static_step"
        myModel.StaticStep(name=step_name, previous="Initial", timePeriod=1.0, nlgeom=ON, initialInc=0.01,
                           minInc=1e-5, maxInc=0.1, maxNumInc=1000, description="Plasticity analysis step")
    else:
        raise ValueError("Solver not supported")

    # field output

    # consider 52 points below since we include 0 and one additional strain step at the end which we remove later
    myModel.TimePoint(name='TimePoints-2', points=(
        (0.0, ), (0.13431, ), (0.17291, ), (0.20117,
                                            ), (0.22446, ), (0.24473, ), (0.26295, ), (0.27968, ),
        (0.29526, ), (0.30994, ), (0.32388, ), (0.33723,
                                                ), (0.35008, ), (0.36251, ), (0.37458, ), (0.38634, ),
        (0.39785, ), (0.40913, ), (0.42023, ), (0.43116,
                                                ), (0.44196, ), (0.45266, ), (0.46327, ), (0.47381, ),
        (0.4843, ), (0.49477, ), (0.50523, ), (0.5157,
                                               ), (0.52619, ), (0.53673, ), (0.54734, ), (0.55804, ),
        (0.56884, ), (0.57977, ), (0.59087, ), (0.60215,
                                                ), (0.61366, ), (0.62542, ), (0.63749, ), (0.64992, ),
        (0.66277, ), (0.67612, ), (0.69006, ), (0.70474,
                                                ), (0.72032, ), (0.73705, ), (0.75527, ), (0.77554, ),
        (0.79883, ), (0.82709, ), (0.86569, ), (1.0, ))
    )

    myModel.fieldOutputRequests['F-Output-1'].setValues(
        timePoint='TimePoints-2', variables=('S', 'PE', 'PEEQ', 'U', 'RF', 'COORD'))

    # establish displacement and reaction force recording
    del myModel.historyOutputRequests['H-Output-1']
    myModel.HistoryOutputRequest(createStepName=step_name, name='H-Output-0', rebar=EXCLUDE, region=root_assem.sets['TopEdgeSet'], sectionPoints=DEFAULT,
                                 variables=('U2', 'RF2'), timePoint='TimePoints-2')

    if solver == 'explicit':
        myModel.HistoryOutputRequest(createStepName=step_name, name='H-Output-1', timePoint='TimePoints-2',
                                     variables=('ALLAE', 'ALLKE', 'ALLIE', 'ETOTAL'))
    else:
        myModel.HistoryOutputRequest(createStepName=step_name, name='H-Output-1', timePoint='TimePoints-2',
                                     variables=('ALLAE', 'ALLSD', 'ALLKE', 'ALLIE', 'ETOTAL'))
    # ### BCs ICs

    # establish smooth amplitude
    amp_name = "Amp-1"
    myModel.SmoothStepAmplitude(
        data=((0.0, 0.0), (1.0, 1.0)), name=amp_name, timeSpan=STEP)
    myModel.DisplacementBC(name="bot_uy", createStepName="Initial",
                           region=root_assem.sets["BottomEdgeSet"], u1=UNSET, u2=SET, ur3=UNSET)
    myModel.DisplacementBC(amplitude=amp_name, name="top_uy", createStepName=step_name,
                           region=root_assem.sets["TopEdgeSet"], u1=UNSET, u2=strain, ur3=UNSET)

    # arbitrary node for rm RBM
    root_assem.Set(name='anchor_node', nodes=allNodes[:1])
    myModel.DisplacementBC(name="anchor_ux", createStepName="Initial",
                           region=root_assem.sets['anchor_node'], u1=SET, u2=UNSET, ur3=UNSET)

    # apply periodic boundary conditions
    periodic_bc_flag = True
    delta = 1.e-3
    for idx, node in enumerate(allNodes):
        # and (not np.isclose(node.coordinates[1], 0.)) and (not np.isclose(node.coordinates[1], 1.))):
        if np.isclose(node.coordinates[0], 0.):
            cur_node_lr = allNodes.getByBoundingBox(
                -delta, node.coordinates[1]-delta, 0.-delta, +delta, node.coordinates[1]+delta, 0. + delta)
            cur_node_lr_partner = allNodes.getByBoundingBox(
                1. - delta, node.coordinates[1]-delta, 0.-delta, 1. + delta, node.coordinates[1]+delta, 0. + delta)
            if not cur_node_lr_partner:
                periodic_bc_flag = False
            root_assem.Set(name='BC_lr_' + str(idx) + 'A', nodes=cur_node_lr)
            root_assem.Set(name='BC_lr_' + str(idx) +
                           'B', nodes=cur_node_lr_partner)
            # apply periodic boundary conditions in x direction (but both for x and y displacements)
            myModel.Equation(name='pbc-' + str(idx) + '_lr_x', terms=(
                (1.0, 'BC_lr_' + str(idx) + 'A', 1), (-1.0, 'BC_lr_' + str(idx) + 'B', 1)))
            myModel.Equation(name='pbc-' + str(idx) + '_lr_y', terms=(
                (1.0, 'BC_lr_' + str(idx) + 'A', 2), (-1.0, 'BC_lr_' + str(idx) + 'B', 2)))
        # and (not np.isclose(node.coordinates[0], 0.)) and (not np.isclose(node.coordinates[0], 1.))):
        elif np.isclose(node.coordinates[1], 0.):
            cur_node_ud = allNodes.getByBoundingBox(
                node.coordinates[0]-delta, 0. - delta, 0. - delta, node.coordinates[0] + delta, 0. + delta, 0. + delta)
            cur_node_ud_partner = allNodes.getByBoundingBox(
                node.coordinates[0] - delta, 1. - delta, 0.-delta, node.coordinates[0] + delta, 1. + delta, 0. + delta)
            root_assem.Set(name='BC_ud_' + str(idx) + 'A', nodes=cur_node_ud)
            root_assem.Set(name='BC_ud_' + str(idx) +
                           'B', nodes=cur_node_ud_partner)
            myModel.Equation(name='pbc-' + str(idx) + '_ud_x', terms=(
                (1.0, 'BC_ud_' + str(idx) + 'A', 1), (-1.0, 'BC_ud_' + str(idx) + 'B', 1)))
            if not cur_node_ud_partner:
                periodic_bc_flag = False

    # ### contact
    int_cont_prop_name = "IntProp-1"
    myModel.ContactProperty(int_cont_prop_name)
    myModel.interactionProperties[int_cont_prop_name].TangentialBehavior(
        dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None,
        formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION,
        pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF,
        table=((0.4, ), ), temperatureDependency=OFF)
    myModel.interactionProperties[int_cont_prop_name].NormalBehavior(
        allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
        pressureOverclosure=HARD)

    uc_surf_name = "uc_surf"
    UCPart.Surface(name=uc_surf_name, side1Edges=UCPart.edges)

    self_cont_name = "SelfContact-1"

    if solver == 'explicit':
        myModel.SelfContactExp(createStepName=step_name, interactionProperty=int_cont_prop_name,
                               mechanicalConstraint=KINEMATIC, name=self_cont_name, surface=uc_assem_inst.surfaces[uc_surf_name], thickness=ON)
    elif solver == 'implicit' or solver == 'static':
        myModel.SelfContactStd(createStepName=step_name, interactionProperty=int_cont_prop_name,
                               name=self_cont_name, surface=uc_assem_inst.surfaces[uc_surf_name], thickness=ON)
    else:
        raise ValueError("Solver not supported")

    # ###

    job_name = 'MyJob'
    # create job
    mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF,
            description='', echoPrint=OFF, explicitPrecision=SINGLE, historyPrint=OFF,
            memory=90, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF,
            multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE,
            numCpus=1, numDomains=1, parallelizationMethodExplicit=DOMAIN, queue=None,
            resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=0, waitMinutes=0)

    # save inp file
    mdb.jobs[job_name].writeInput(consistencyChecking=OFF)


# %%

file_base = "/work/nvme/bbka/qibang/repository_WNbbka/TRAINING_DATA/Geo2DReduced/abaqus/femDataR1"

sec_id_start, sec_id_end = 20, 30


for sec_id in range(sec_id_start, sec_id_end):
    file_pre = os.path.join(file_base, f"sec_{sec_id}")
    # working_dirs=sorted([os.path.join(file_pre, f.name) for f in os.scandir(file_pre) if f.is_dir()])
    sample_idxs = [int(f.name.split('_')[1])
                   for f in os.scandir(file_pre) if f.is_dir()]
    sample_idxs.sort()
    working_dirs = [os.path.join(
        file_pre, f"sample_{idx}") for idx in sample_idxs]
    for working_dir in working_dirs:
        inp_file = os.path.join(working_dir, "MyJob.inp")
        rpy_file = os.path.join(working_dir, "abaqus.rpy")
        if not os.path.exists(inp_file):
            try:
                generate_inp(working_dir)
                print(f"Generating inp file: {working_dir}")
            except Exception as e:
                print(f"Error in generating inp file: {working_dir}")
