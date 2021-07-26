import numpy as np
import time
import h5py

# Constants for unit conversion
CM_TO_MPC = 3.085678e24
CMPERS_TO_KMPERS = 1e5
MSUN_TO_GRAM = 1.989e33

def convSFToMTF(startSnap, endSnap, fieldsDict, fn, basehalo_fn):

    redshift, MTFdata = LoadSFIntoMTF(startSnap, endSnap, fieldsDict, fn, basehalo_fn)

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
            groupCatalog[catValue] = np.array(groupData[catValue][:numHalos], dtype = fieldTypes[i])
    
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
            subhaloCatalog[catValue] = np.array(subhaloData[catValue][:numSubhalos], dtype = fieldTypes[i])
    
    return groupCatalog, subhaloCatalog


def LoadSFIntoMTF(startSnap, endSnap, fieldsDict, fn, basehalo_fn, HALOIDVAL = 1000000000000):

    treeFields = ["HaloID", "StartProgenitor", "Progenitor", "Descendant", "EndDescendant", "HostHaloID"]
    otherFields = [field for field in fieldsDict.keys() if field not in treeFields]
    MTFfieldnames = treeFields + otherFields

    start = time.time()

    MTFdata = {"Snap_%03d" %snap:{field:[] for field in MTFfieldnames} for snap in range(startSnap, endSnap + 1)}

    hf = h5py.File(fn, 'r')
    snapnum = hf['TreeHalos/SnapNum'][()]
    redshift = hf['TreeTimes/Redshift'][()]
    tree_id = hf['TreeHalos']['TreeID'][()]
    sh_id = hf['TreeHalos']['SubhaloNr'][()]
    tree_main_prog = hf['TreeHalos']['TreeMainProgenitor'][()]
    tree_desc = hf['TreeHalos']['TreeDescendant'][()]

    hf.close()

    for isnap in range(startSnap, endSnap + 1):
        
        snapKey = 'Snap_%03d' % isnap
        snap_inds = np.where(snapnum == isnap)[0]
        numhalos = snap_inds.size

        halocatfile = basehalo_fn + 'fof_subhalo_tab_%03d.hdf5' % isnap

        f = h5py.File(halocatfile, 'r')

        nfof = f['Header'].attrs['Ngroups_ThisFile']
        nsub = f['Header'].attrs['Nsubhalos_ThisFile']
        redshift[isnap] = f['Header'].attrs['Redshift']
        a = 1 / (1 + redshift[isnap])
        h = f['Parameters'].attrs['HubbleParam']
        length_unit = f['Parameters'].attrs['UnitLength_in_cm']
        mass_unit = f['Parameters'].attrs['UnitMass_in_g']
        vel_unit = f['Parameters'].attrs['UnitVelocity_in_cm_per_s']
        fof_group = f['Group']
        sub_group = f['Subhalo']

        # Simple (sub)halo properties
        # Mass in units of [10^10 Msun]
        # Velocity in units of [pkm / s]
        # Length in units of [cMpc]
        mass = sub_group['SubhaloMass'][()] / h * mass_unit / (1e10 * MSUN_TO_GRAM)
        pos = sub_group['SubhaloPos'][()] / h * length_unit / CM_TO_MPC
        vel = sub_group['SubhaloVel'][()] * a**0.5 * vel_unit / CMPERS_TO_KMPERS
        rad = sub_group['SubhaloHalfmassRad'][()] / h * length_unit / CM_TO_MPC

        # Host halo ID for each (sub)halo (-1 if it has no host)
        # 1-index like VELOCIraptor
        # Central subhaloes treated as FOF groups
        hostHaloID = fof_group['GroupFirstSub'][()][sub_group['SubhaloGroupNr'][()]] + 1
        subRank = sub_group['SubhaloRankInGr'][()]
        hostHaloID[np.where(subRank == 0)] = -1
        
        # Tree-related fields (i.e. progenitors and descendants)
        for ihalo in range(numhalos):

            #print('***** HALO ', ihalo, ' *****')

            # Find current halo in tree
            ind = np.where((snapnum == isnap) & (sh_id == ihalo))[0][0]
            treeID = tree_id[ind]
            tree_inds = np.where(tree_id == treeID)[0]

            # Locate progenitor
            if isnap > startSnap:
                prog_ind = tree_main_prog[ind]
                progSnap = snapnum[tree_inds[prog_ind]]
                prog = sh_id[tree_inds[prog_ind]]
                if prog_ind == -1:
                    MTFdata[snapKey]['Progenitor'].append(isnap * HALOIDVAL + ihalo + 1)
                else:
                    progID = progSnap * HALOIDVAL + prog + 1
                    MTFdata[snapKey]['Progenitor'].append(progID)
                MTFdata[snapKey]['StartProgenitor'].append(-99)
            
            # Locate descendant
            if isnap < endSnap:
                desc_ind = tree_desc[ind]
                descSnap = snapnum[tree_inds[desc_ind]]
                desc = sh_id[tree_inds[desc_ind]]
                if desc_ind == -1:
                    MTFdata[snapKey]['Descendant'].append(isnap * HALOIDVAL + ihalo + 1)
                else:
                    descID = descSnap * HALOIDVAL + desc + 1
                    MTFdata[snapKey]['Descendant'].append(descID)
                MTFdata[snapKey]['EndDescendant'].append(-99)

            # (Sub)halo properties
            MTFdata[snapKey]['Mass'].append(mass[ihalo])
            MTFdata[snapKey]['Pos'].append(pos[ihalo])
            MTFdata[snapKey]['Vel'].append(vel[ihalo])
            MTFdata[snapKey]['Radius'].append(rad[ihalo])

            # (Sub)halo IDs and host halo IDs
            MTFdata[snapKey]['HaloID'].append(isnap * HALOIDVAL + ihalo + 1)
            MTFdata[snapKey]['HostHaloID'].append(hostHaloID[ihalo])

            # Set progenitors, start progenitors, descendants, and end descendants at first and last snapshots
            if isnap == startSnap:
                MTFdata[snapKey]['Progenitor'].append(isnap * HALOIDVAL + ihalo + 1)
                MTFdata[snapKey]['StartProgenitor'].append(isnap * HALOIDVAL + ihalo + 1)
            if isnap == endSnap:
                MTFdata[snapKey]['Descendant'].append(isnap * HALOIDVAL + ihalo + 1)
                MTFdata[snapKey]['EndDescendant'].append(isnap * HALOIDVAL + ihalo + 1)

        # Close the (sub)halo catalog file
        f.close()


    # Convert everything into a numpy array for easy indexing
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
            
            if(MTFdata[snapKey]['StartProgenitor'][ihalo] == -99): # This should always be the case except for first snapshot

                haloID = MTFdata[snapKey]['HaloID'][ihalo]
                progID = MTFdata[snapKey]['Progenitor'][ihalo]
                progSnap = int(progID / HALOIDVAL)
                progSnapKey = 'Snap_%03d' % progSnap
                progIndex = int(progID % HALOIDVAL - 1)
                
                if haloID == progID: # We're at the start of a branch
                    MTFdata[snapKey]['StartProgenitor'][ihalo] = haloID
                else:
                    tmpStartProgenitor = MTFdata[progSnapKey]['StartProgenitor'][progIndex]
                    MTFdata[snapKey]['StartProgenitor'][ihalo] = tmpStartProgenitor
                    
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
            
            if(MTFdata[snapKey]['EndDescendant'][ihalo] == -99): # This should always be the case except for final snapshot
                
                haloID = MTFdata[snapKey]['HaloID'][ihalo]
                descID = MTFdata[snapKey]['Descendant'][ihalo]
                descSnap = int(descID / HALOIDVAL)
                descSnapKey = 'Snap_%03d' % descSnap
                descIndex = int(descID % HALOIDVAL - 1)
                
                if haloID == descID: # We're at the end of a branch
                    MTFdata[snapKey]['EndDescendant'][ihalo] = haloID
                else:
                    tmpEndDescendant = MTFdata[descSnapKey]['EndDescendant'][descIndex]
                    MTFdata[snapKey]['EndDescendant'][ihalo] = tmpEndDescendant
        
        print("Done snap", snap, "in", time.time() - start)
                
    print("Done setting EndDescendants in", time.time() - totstart)

    return MTFdata
                

