import numpy as np
import time
import h5py

def convSFToMTF(startSnap, endSnap, fieldsDict, fn):

	redshift, MTFdata = LoadSFIntoMTF(startSnap, endSnap, fieldsDict, fn)

	# Set StartProgenitors and EndDescendants
	MTFdata = convToMTF(startSnap, endSnap, MTFdata)

	return redshift, MTFdata


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

	for snap in range(startSnap, endSnap + 1):
	    
	    snapKey = 'Snap_%03d' % snap
	    isnap = snap - startSnap
	    snap_inds = np.where(snapnum == isnap)[0]
	    numhalos = snap_inds.size

	    # Group IDs for each subhalo in isnap
	    indexes = np.unique(host_id[snap_inds], return_index = True)[1]
	    host_ids = np.array([host_id[snap_inds][index] for index in sorted(indexes)])
	    
	    # Tree-related fields (i.e. progenitors and descendants)
	    progSnap = isnap - 1
	    descSnap = isnap + 1
	    for ihalo in range(numhalos):

	    	# Temporally unique HaloID
	    	MTFdata[snapKey]['HaloID'].append(isnap * HALOIDVAL + len(MTFdata[snapKey]['HaloID']) + 1)
	    	host_ind = np.where(host_ids == host_id[snap_inds][ihalo])[0][0] # Index of group halo in isnap
	    	MTFdata[snapKey]['HostHaloID'].append(isnap * HALOIDVAL + host_ind + 1)

	    	# If HaloID = HostHaloID then we are at a central subhalo, which represents the group in SF - set HostHaloID = -1
	    	if(MTFdata[snapKey]['HaloID'][ihalo] == MTFdata[snapKey]['HostHaloID'][ihalo]):
	    		MTFdata[snapKey]['HostHaloID'][ihalo] = -1

	    	# Locate progenitors and descendants
	    	tid = tree_id[snap_inds[ihalo]] # ID of the tree this halo is in
	    	pind = tree_main_prog[snap_inds[ihalo]] # Index within this tree of the progenitor
	    	dind = tree_desc[snap_inds[ihalo]] # Index within this tree of the descendant
	    	pid = np.where(np.where((tree_id == tid) & (snapnum == progSnap))[0] == pind)[0] # Progenitor index in progSnap
	    	did = np.where(np.where((tree_id == tid) & (snapnum == descSnap))[0] == dind)[0] # Descendant index in descSnap

	    	if pid.size == 0: # We've hit the head (root) of a branch
	    		pid = -1
	    		MTFdata[snapKey]['Progenitor'].append(MTFdata[snapKey]['HaloID'][ihalo])
	    	else:
	    		progID = pid + 1 + progSnap * HALOIDVAL
	    		MTFdata[snapKey]['Progenitor'].append(progID[0])
	    	if did.size == 0: # We've hit the tail (end) of a branch
	    		did = -1
	    		MTFdata[snapKey]['Descendant'].append(MTFdata[snapKey]['HaloID'][ihalo])
	    	else:
	    		descID = did + 1 + descSnap * HALOIDVAL
	    		MTFdata[snapKey]['Descendant'].append(descID[0])

	    	MTFdata[snapKey]['StartProgenitor'].append(0)
	    	MTFdata[snapKey]['EndDescendant'].append(0)
	    
	    
	    # Halo property fields (e.g. position, velocity, mass, radius)
	    for field in otherFields:
	    	fieldValue = fieldsDict[field][0]
	    	MTFdata[snapKey][field] = hf['TreeHalos'][fieldValue][()][snap_inds]

	    	# Convert to appropriate units (comoving length, physical vel, Mpc, km/s, 10^10 Msun)
	    	params = hf['Parameters'].attrs
	    	hubble = params['HubbleParam']
	    	cm_to_Mpc = 3.085678e24
	    	cmpers_to_kmpers = 1e5
	    	msun_to_gram = 1.989e33

	    	if field == 'Pos': # Convert to cMpc
	    		MTFdata[snapKey][field] *= 1 / hubble * params['UnitLength_in_cm'] / cm_to_Mpc
	    	elif field[0] == 'V': # Convert to pkm/s
	    		MTFdata[snapKey][field] *= scale_fac[isnap]**0.5 * params['UnitVelocity_in_cm_per_s'] / cmpers_to_kmpers
	    	elif field[0] == 'M': # Convert to 10^10Msun
	    		MTFdata[snapKey][field] *= 1 / hubble * params['UnitMass_in_g'] / (1e10 * msun_to_gram)
	    	elif field[0] == 'R': # Convert to cMpc
	    		MTFdata[snapKey][field] *= 1 / hubble * params['UnitLength_in_cm'] / cm_to_Mpc

	hf.close()

	#Convert everything into array for easy indexing
	for snap in range(startSnap, endSnap+1):
		snapKey = 'Snap_%03d' % snap
		for field in MTFdata[snapKey].keys():
			MTFdata[snapKey][field] = np.asarray(MTFdata[snapKey][field])

	print("Done loading the data into EFT format in",time.time()-start)

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
                

