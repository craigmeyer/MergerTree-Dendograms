import numpy as np
import time
import h5py

def convSFToMTF(startSnap, endSnap, fieldsDict, fn):

	redshift, MTFdata = LoadSFIntoMTF(startSnap, endSnap, fieldsDict, fn)

	# Set StartProgenitors and EndDescendants
	MTFdata = convToMTF(startSnap, endSnap, MTFdata)

	return redshift, MTFdata

def readSFHaloProperties(filename, desiredGroupFields, desiredSubhaloFields):
    
    halofile = h5py.File(filename, 'r')
    
    # Read in header
    # NOTE: Assumes only one file per snapshot
    # todo: Expand functionality to handle snapshots split over multiple files
    header = halofile['Header'].attrs
    numHalos = np.uint64(header['Ngroups_ThisFile'])
    numSubhalos = header['Nsubhalos_ThisFile']
    
    # Read in FOF group data
    groupData = halofile['Group']
    fieldNames = []
    fieldTypes = []
    
    # Load only the specified fields if any are passed
    if len(desiredGroupFields) > 0:
        fieldNames = desiredGroupFields
        fieldTypes = [groupData[field].dtype for field in fieldNames]
    else:
        fieldNames = [str(key) for key in groupData.keys()]
        fieldTypes = [groupData[field].dtype for field in fieldNames]
    numFields = len(fieldNames)
    
    # Create group dictionary
    groupCatalog = {}
    for i in range(numFields):
        catValue = fieldNames[i]
        if numHalos > 0:
            groupCatalog[catValue] = [np.array(groupData[catValue][:numHalos])]
    
    # Read in subhalo data
    subhaloData = halofile['Subhalo']
    fieldNames = []
    fieldTypes = []
    
    # Load only the specified fields if any are passed
    if len(desiredSubhaloFields) > 0:
        fieldNames = desiredSubhaloFields
        fieldTypes = [subhaloData[field].dtype for field in fieldNames]
    else:
        fieldNames = [str(key) for key in subhaloData.keys()]
        fieldTypes = [subhaloData[field].dtype for field in fieldNames]
    numFields = len(fieldNames)
    
    # Create subhalo dictionary
    subhaloCatalog = {}
    for i in range(numFields):
        catValue = fieldNames[i]
        if numSubhalos > 0:
            subhaloCatalog[catValue] = [np.array(subhaloData[catValue][:numSubhalos])]
    
    return groupCatalog, subhaloCatalog


def LoadSFIntoMTF(startSnap, endSnap, fieldsDict, fn, HALOIDVAL = 1000000000000):

	treeFields = ["HaloID", "StartProgenitor", "Progenitor", "Descendant", "EndDescendant", "HostHaloID"]
	otherFields = [field for field in fieldsDict.keys() if field not in treeFields]
	MTFfieldnames = treeFields + otherFields

	start = time.time()

	MTFdata = {"Snap_%03d" %snap:{field:[] for field in MTFfieldnames} for snap in range(startSnap, endSnap + 1)}

	hf = h5py.File(fn, 'r')
	snapnum = hf['TreeHalos/SnapNum'][()]
	redshift = hf['TreeTimes/Redshift'][()]
	scale_fac = 1 / (1 + redshift)
	host_id = hf['TreeHalos']['GroupNr'][()]
	sh_id = hf['TreeHalos']['SubhaloNr'][()]
	tree_id = hf['TreeHalos']['TreeID'][()]
	tree_index = hf['TreeHalos']['TreeIndex'][()]
	tree_main_prog = hf['TreeHalos']['TreeMainProgenitor'][()]
	tree_desc = hf['TreeHalos']['TreeDescendant'][()]

	params = hf['Parameters'].attrs
	hubble = params['HubbleParam']
	length_unit = params['UnitLength_in_cm']
	vel_unit = params['UnitVelocity_in_cm_per_s']
	mass_unit = params['UnitMass_in_g']
	cm_to_Mpc = 3.085678e24
	cmpers_to_kmpers = 1e5
	msun_to_gram = 1.989e33

	hf.close()

	for snap in range(startSnap, endSnap + 1):
	    
	    snapKey = 'Snap_%03d' % snap
	    isnap = snap - startSnap
	    snap_inds = np.where(snapnum == isnap)[0]
	    numhalos = snap_inds.size

	    catfilename = '/Users/craigmeyer/Documents/PhD/VELOCIraptor/L20N128/050/SF_HBT/fof_subhalo_tab_%03d.hdf5' % snap
	    groupdata, subdata = readSFHaloProperties(catfilename, [], []) # Read in all fields

	    # # Group IDs for each subhalo in isnap
	    # indexes = np.unique(host_id[snap_inds], return_index = True)[1]
	    # host_ids = np.array([host_id[snap_inds][index] for index in sorted(indexes)])
	    MTFdata[snapKey]['HostHaloID'] = host_id[snap_inds]
	    
	    # Tree-related fields (i.e. progenitors and descendants)
	    progSnap = isnap - 1
	    descSnap = isnap + 1
	    for ihalo in range(numhalos):

	    	# Temporally unique HaloID
	    	MTFdata[snapKey]['HaloID'].append(isnap * HALOIDVAL + len(MTFdata[snapKey]['HaloID']) + 1)
	    	# host_ind = np.where(host_ids == host_id[snap_inds][ihalo])[0][0] # Index of group halo in isnap
	    	# MTFdata[snapKey]['HostHaloID'].append(isnap * HALOIDVAL + host_ind + 1)

	    	# If HaloID = HostHaloID then we are at a central subhalo, which represents the group in SF - set HostHaloID = -1
	    	# Set HostHaloID to -1 for a central subhalo, which represents the group in SF
	    	if subdata['SubhaloRankInGr'][sh_id[snap_inds][ihalo]] == 0:
	    		MTFdata[snapKey]['HostHaloID'][ihalo] = -1

	    	# Locate progenitors and descendants
	    	tid = tree_id[snap_inds[ihalo]] # ID of the branch this halo is in
	    	pind = tree_main_prog[snap_inds[ihalo]] # Index within this branch of the progenitor
	    	dind = tree_desc[snap_inds[ihalo]] # Index within this branch of the descendant
	    	pid = np.where(np.where((tree_id == tid) & (snapnum == progSnap))[0] == pind)[0] # Progenitor index in progSnap
	    	did = np.where(np.where((tree_id == tid) & (snapnum == descSnap))[0] == dind)[0] # Descendant index in descSnap

	    	if pid.size == 0: # We've hit the tail (start) of a branch
	    		pid = -1
	    		MTFdata[snapKey]['Progenitor'].append(MTFdata[snapKey]['HaloID'][ihalo])
	    	else:
	    		progID = pid + 1 + progSnap * HALOIDVAL
	    		MTFdata[snapKey]['Progenitor'].append(progID[0])
	    	if did.size == 0: # We've hit the head (end) of a branch
	    		did = -1
	    		MTFdata[snapKey]['Descendant'].append(MTFdata[snapKey]['HaloID'][ihalo])
	    	else:
	    		descID = did + 1 + descSnap * HALOIDVAL
	    		MTFdata[snapKey]['Descendant'].append(descID[0])

	    	MTFdata[snapKey]['StartProgenitor'].append(0)
	    	MTFdata[snapKey]['EndDescendant'].append(0)

	    # Halo property fields
	    for field in otherFields:
	    	MTFdata[snapKey][field] = subdata[field][sh_id[snap_inds]]

	    	# Convert fields to appropraite units (cMpc, pkm/s, 10^10Msun)
	    	if field[0] == 'P': # Pos field, convert to cMpc
	    		MTFdata[snapKey][field] *= 1 / hubble * length_unit / cm_to_Mpc
	    	elif field[0] == 'V': # Convert to pkm/s
	    		MTFdata[snapKey][field] *= scale_fac[isnap]**0.5 * vel_unit / cmpers_to_kmpers
	    	elif field[0] == 'M': # Convert to 10^10 Msun
	    		MTFdata[snapKey][field] *= 1 / hubble * mass_unit / (1e10 * msun_to_gram)
	    	elif field[0] == 'R': # Convert to cMpc
	    		MTFdata[snapKey][field] *= 1 / hubble * length_unit / cm_to_Mpc




	    # # Halo property fields (e.g. position, velocity, mass, radius)
	    # for field in otherFields:
	    # 	fieldValue = fieldsDict[field][0]
	    # 	MTFdata[snapKey][field] = hf['TreeHalos'][fieldValue][()][snap_inds]

	    # 	# Convert to appropriate units (comoving length, physical vel, Mpc, km/s, 10^10 Msun)
	    # 	params = hf['Parameters'].attrs
	    # 	hubble = params['HubbleParam']
	    # 	cm_to_Mpc = 3.085678e24
	    # 	cmpers_to_kmpers = 1e5
	    # 	msun_to_gram = 1.989e33

	    # 	if field == 'Pos': # Convert to cMpc
	    # 		MTFdata[snapKey][field] *= 1 / hubble * params['UnitLength_in_cm'] / cm_to_Mpc
	    # 	elif field[0] == 'V': # Convert to pkm/s
	    # 		MTFdata[snapKey][field] *= scale_fac[isnap]**0.5 * params['UnitVelocity_in_cm_per_s'] / cmpers_to_kmpers
	    # 	elif field[0] == 'M': # Convert to 10^10Msun
	    # 		MTFdata[snapKey][field] *= 1 / hubble * params['UnitMass_in_g'] / (1e10 * msun_to_gram)
	    # 	elif field[0] == 'R': # Convert to cMpc
	    # 		MTFdata[snapKey][field] *= 1 / hubble * params['UnitLength_in_cm'] / cm_to_Mpc



	#Convert everything into array for easy indexing
	for snap in range(startSnap, endSnap + 1):
		snapKey = 'Snap_%03d' % snap
		for field in MTFdata[snapKey].keys():
			MTFdata[snapKey][field] = np.asarray(MTFdata[snapKey][field])

	print("Done loading the data into EFT format in", time.time() - start)

	return redshift, MTFdata


def convToMTF(startSnap, endSnap, MTFdata, HALOIDVAL = 1000000000000):
    
    # Setting start progenitors
    totstart = time.time()
    print("Setting StartProgenitors")

    for snap in range(startSnap, endSnap + 1):
        
        start = time.time()
        
        snapKey = 'Snap_%03d' % snap
        isnap = snap - startSnap
        numhalos = len(MTFdata[snapKey]['HaloID'])
        
        for ihalo in range(numhalos):
            
            if(MTFdata[snapKey]['StartProgenitor'][ihalo] == 0): # This should always be the case
                
                haloID = MTFdata[snapKey]['HaloID'][ihalo]
                branchStartProgenitor = haloID
                MTFdata[snapKey]['StartProgenitor'][ihalo] = branchStartProgenitor
                
                # Get descendant information
                descID = MTFdata[snapKey]['Descendant'][ihalo]
                descSnap = int(descID / HALOIDVAL)
                descSnapKey = 'Snap_%03d' % descSnap
                descIndex = int(descID % HALOIDVAL - 1)
                descProg = MTFdata[descSnapKey]['Progenitor'][descIndex]
                
                # Check we haven't reached the tail (end) of the branch
                while((descID != haloID) & (descProg == haloID)):
                    
                    # Move down the branch, setting the start progenitor for each descendant halo
                    haloID = descID
                    haloSnap = descSnap
                    haloSnapKey = descSnapKey
                    haloIndex = descIndex
                    MTFdata[haloSnapKey]['StartProgenitor'][haloIndex] = branchStartProgenitor
                    
                    # Get descendant information
                    descID = MTFdata[haloSnapKey]['Descendant'][haloIndex]
                    descSnap = int(descID / HALOIDVAL)
                    descSnapKey = 'Snap_%03d' % descSnap
                    descIndex = int(descID % HALOIDVAL - 1)
                    descProg = MTFdata[descSnapKey]['Progenitor'][descIndex]
                    
        print("Done snap",snap,"in",time.time()-start)
        
    print("Done setting StartProgenitors in", time.time() - totstart)
        
    # Setting end descendants
    totstart = time.time()
    print("Setting EndDescendants")
    
    for snap in range(endSnap, startSnap - 1, -1):
        
        start = time.time()
        
        snapKey = 'Snap_%03d' % snap
        isnap = snap - startSnap
        numhalos = len(MTFdata[snapKey]['HaloID'])
        
        for ihalo in range(numhalos):
            
            if(MTFdata[snapKey]['EndDescendant'][ihalo] == 0): # This should always be the case
                
                haloID = MTFdata[snapKey]['HaloID'][ihalo]
                branchEndDescendant = haloID
                MTFdata[snapKey]['EndDescendant'][ihalo] = branchEndDescendant
                
                # Get progenitor information
                progID = MTFdata[snapKey]['Progenitor'][ihalo]
                progSnap = int(progID / HALOIDVAL)
                progSnapKey = 'Snap_%03d' % progSnap
                progIndex = int(progID % HALOIDVAL - 1)
                progDesc = MTFdata[progSnapKey]['Descendant'][progIndex]
                
                # Check we haven't reached the head (root) of the branch
                while((progID != haloID) & (progDesc == haloID)):
                    
                    # Move up the branch, setting the end descendant for each progenitor halo
                    haloID = progID
                    haloSnap = progSnap
                    haloSnapKey = progSnapKey
                    haloIndex = progIndex
                    MTFdata[haloSnapKey]['EndDescendant'][haloIndex] = branchEndDescendant
                    
                    # Get progenitor information
                    progID = MTFdata[haloSnapKey]['Progenitor'][haloIndex]
                    progSnap = int(progID / HALOIDVAL)
                    progSnapKey = 'Snap_%03d' % progSnap
                    progIndex = int(progID % HALOIDVAL - 1)
                    progDesc = MTFdata[progSnapKey]['Descendant'][progIndex]
        
        print("Done snap", snap, "in", time.time() - start)
                
    print("Done setting EndDescendants in", time.time() - totstart)

    return MTFdata
                

