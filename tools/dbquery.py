import sqlite3
import pdb
import numpy as np


def bird_dict_genetic_parents(birds) -> dict:
    con = sqlite3.connect("db.sqlite3")
    cur = con.cursor()

    gen_parents = {}
    for bird in birds:
        try:
            uuid, exists = getUUIDfromBands(cur,bird)
        except sqlite3.OperationalError:
            UserWarning('debug sql code')
            gen_parents[bird] = 'fix me annie'
            return gen_parents
        if exists:
            (parent1_uuid, parent2_uuid) = getGeneticParentsfromUUID(cur, uuid)
            if (parent1_uuid == 'NULL') and parent2_uuid == 'NULL':
                gen_parents[bird] = 'unlabelled in db'
            else:
                try:
                    parent1 = getBandsfromUUID(cur, parent1_uuid[0])
                except Exception:  #sqlite3.OperationalError:
                    parent1 = '?'
                try:
                    parent2 = getBandsfromUUID(cur, parent2_uuid[0])
                except Exception:  #sqlite3.OperationalError:
                    parent2 = '?'
                gen_parents[bird] = parent1 + '_' + parent2
        else:
            gen_parents[bird] = 'unlabelled in db'
            print('bird %s has no uuid' % bird)
    return gen_parents

# def get_bird_parent_array(bird_list, type: str = 'nest') -> np.ndarray:
#     con = sqlite3.connect("db.sqlite3")
#     cur = con.cursor()
#     if type == 'genetic':
#         retrieval_func = getGeneticParentsfromUUID
#     elif type == 'nest':
#         retrieval_func = getParentsfromUUID
#
#     bird_parent_array = np.empty((len(bird_list), 3), dtype=object)
#     for i, bird in enumerate(bird_list):
#         try:
#             uuid, exists = getUUIDfromBands(cur,bird)
#         except sqlite3.OperationalError:
#             UserWarning('debug sql code')
#             bird_list[i, 0] = bird
#             bird_list[i, 1] = 'fix me annie'
#             bird_list[i, 2] = 'fix me annie'
#             return bird_parent_array
#         if exists:
#             (parent1_uuid, parent2_uuid) = retrieval_func(cur, uuid)
#             if parent1_uuid == 'NULL':
#                 bird_parent_array[i, 0] = bird
#                 bird_parent_array[i, 1] = 'unlabelled in db'
#                 bird_parent_array[i, 2] = 'unlablled in db'
#             else:
#                 parent1 = getBandsfromUUID(cur, parent1_uuid[0])
#                 parent2 = getBandsfromUUID(cur, parent2_uuid[0])
#                 bird_parent_array[i, 0] = bird
#                 bird_parent_array[i, 1] = parent1
#                 bird_parent_array[i, 2] = parent2
#         else:
#             bird_parent_array[i, 0] = bird + '_no_uuid'
#             bird_parent_array[i, 1] = 'unlabelled in db'
#             bird_parent_array[i, 2] = 'unlablled in db'
#             print('bird %s has no uuid' % bird)
#     return bird_parent_array


def bird_dict_nest_parents(birds):
    con = sqlite3.connect("db.sqlite3")
    cur = con.cursor()

    nest_parents = {}
    for bird in birds:
        try:
            uuid, exists = getUUIDfromBands(cur, bird)
        except sqlite3.OperationalError:
            UserWarning('debug sql code')
            nest_parents[bird] = 'fix me annie'
            return nest_parents
        if exists:
            (parent1_uuid, parent2_uuid) = getParentsfromUUID(cur, uuid)
            if (parent1_uuid == 'NULL') and parent2_uuid == 'NULL':
                nest_parents[bird] = 'unlabelled in db'
            else:
                try:
                    parent1 = getBandsfromUUID(cur, parent1_uuid[0])
                except Exception:# sqlite3.OperationalError:
                    parent1 = '?'
                try:
                    parent2 = getBandsfromUUID(cur, parent2_uuid[0])
                except Exception:# sqlite3.OperationalError:
                    parent2 = '?'
            nest_parents[bird] = parent1 + '_' + parent2
        else:
            nest_parents[bird] = 'unlabelled in db'
            print('bird %s has no uuid' % bird)
    return nest_parents


# Haymish code start below
def getAllBirdsFromDB(cur):
    QUERY="""
    WITH group1 AS (
      SELECT uuid, birds_color.abbrv, birds_animal.band_number
        FROM birds_color INNER JOIN birds_animal ON birds_animal.band_color_id=id
    ),
    group2 AS (
      SELECT uuid, birds_color.abbrv, birds_animal.band_number2
        FROM birds_color INNER JOIN birds_animal ON birds_animal.band_color2_id=id 
    )
    SELECT *
      FROM group1
      JOIN group2 ON group1.uuid = group2.uuid
    ;"""

    output=cur.execute(QUERY)
    temp=output.fetchall()
    return temp;

def allgeneticbirdstoUUID(cur):
    query='SELECT IND_ID FROM bird_geneticdata'
    res=cur.execute(query)
    temp=res.fetchall();
    tempout=[];
    for name in temp:        
        tempstr=name[0]
        tempstr=tempstr.replace('yw','ye')        
        uuid=getUUIDfromBands(cur,tempstr)
        UUIDandBandText=(uuid,tempstr)
        tempout.append(UUIDandBandText);
    return tempout


def BandStrtoDBBand(bandstr):
     import re
     
     bandcolor2=''
     bandnum2=''     

     t=re.findall(r'([\d.]+)|([^\d.]+)',bandstr)
  
     bandcolor1=t[0][1]
     bandnum1=t[1][0]
     if len(t)>2:
        bandcolor2=t[2][1]
     if len(t)>3:
        bandnum2=t[3][0]
     #pdb.set_trace()        
     if bandcolor1=='b':
        bandcolor1='bk'
     if bandcolor2=='b':
        bandcolor2='bk'
     if bandcolor2=='w':
        bandcolor2='wh'
     if bandcolor1=='w':
        bandcolor1='wh'
     if bandcolor2=='g':
        bandcolor2='gr'
     if bandcolor1=='g':
        bandcolor1='gr'
     if bandcolor2=='y':
        bandcolor2='ye'
     if bandcolor1=='y':
        bandcolor1='ye'        
     if bandcolor2=='r':
        bandcolor2='rd'
     if bandcolor1=='r':
        bandcolor1='rd'
     if bandcolor2=='o':
        bandcolor2='or'
     if bandcolor1=='o':
        bandcolor1='or'         
    # pdb.set_trace()                
     return bandcolor1,bandnum1,bandcolor2,bandnum2
    
def getSex(cur,uuid):   
    query="SELECT sex FROM birds_animal WHERE uuid='"+uuid+"'"
    res=cur.execute(query)
    sex=str(res.fetchone()[0])    
    return sex

#DB index to human readable band numbers
def DBBandstoBandStr(cur,ID1,NUM1,ID2,NUM2):   
    query="SELECT abbrv FROM birds_color WHERE id="+str(ID1)   
    res=cur.execute(query)
    coloridx1=str(res.fetchone()[0])    
    query="SELECT abbrv FROM birds_color WHERE id="+str(ID2)   
    res=cur.execute(query)
    coloridx2=str(res.fetchone()[0])
    BandStr=(coloridx1)+str(NUM1)+coloridx2+str(NUM2)
    return BandStr

def getUUIDfromBands(cur,bandstr):
    bandstr=bandstr.replace('yw','ye') #There've been some choices about yellow.
    bandstr=bandstr.replace('yl','ye')  
    bandcolor1,bandnum1,bandcolor2,bandnum2= BandStrtoDBBand(bandstr)
   # pdb.set_trace()
    query="SELECT id FROM birds_color WHERE abbrv="+"'"+bandcolor1+"'"   
    res=cur.execute(query)
 #   pdb.set_trace()
    try:
        coloridx1=str(res.fetchone()[0])
        if bandcolor2!='':
            query="SELECT id FROM birds_color WHERE abbrv="+"'"+bandcolor2+"'"    
            res=cur.execute(query)      
            coloridx2=str(res.fetchone()[0])
            query=("SELECT uuid FROM birds_animal WHERE birds_animal.band_color_id="+coloridx1+
                   " AND birds_animal.band_color2_id="+coloridx2+" AND birds_animal.band_number="+str(bandnum1)+
                   " AND birds_animal.band_number2="+str(bandnum2));
        else:
            query=("SELECT uuid FROM birds_animal WHERE birds_animal.band_color_id="+coloridx1+
                   " AND birds_animal.band_color2_id IS NULL AND birds_animal.band_number="+str(bandnum1))
        res=cur.execute(query)
        temp=res.fetchall();            
    except:
        return 'nouuid', 0

    
    if len(temp)>0:   
        return temp[0][0], len(temp)
    else:
        #print('not in db')
        return 'nouuid', 0


def getBandsfromUUID(cur,uuid):    
    try:
        query="SELECT band_color_id, band_color2_id, band_number, band_number2 FROM birds_animal WHERE uuid=""'"+uuid+"'"
        res=cur.execute(query)
        indices=(res.fetchall())
        try:
            bndcolor1=indices[0][0]
            bndcolor2=indices[0][1]
            bndnum1=indices[0][2]
            bndnum2=indices[0][3]
        except IndexError:
            # this for some parents where there seem to be issues finding band color
            return 'missing bands'
        BirdName=DBBandstoBandStr(cur,bndcolor1,bndnum1,bndcolor2,bndnum2)
    except Exception as e:
        print(f'Error {e} retrieving parent names')
        return '?'
    return BirdName

def getRearedOffspringfromParent(cur,uuid):
    query="SELECT child_id FROM birds_parent WHERE parent_id='"+uuid+"'"
    res=cur.execute(query)
    temp=res.fetchall();
    return temp;    
    
def getRearedOffspringfromParents(cur,fatheruuid,motheruuid):
    query="""
    WITH group1 AS (
      SELECT child_id FROM birds_parent WHERE parent_id='"""+fatheruuid+"'""""
    ),
    group2 AS (
      SELECT child_id FROM birds_parent WHERE parent_id='"""+motheruuid+"'""""     
    )
    SELECT *
      FROM group1
      JOIN group2 ON group1.child_id = group2.child_id
    ;"""
  
    res=cur.execute(query)
    temp=res.fetchall();    
    return temp

def isParent(cur, uuid):

    query="SELECT parent_id FROM birds_parent WHERE parent_id='"+uuid+"'";
    res=cur.execute(query)
    temp=res.fetchall()

    if len(temp)>0:
        return True
    else:
        return False

def hasNotes(cur, uuid):
    query="SELECT notes FROM birds_animal WHERE uuid='"+uuid+"'";
    res=cur.execute(query)
    temp=res.fetchall()
    
    if len(temp)>0:
        return True
    else:
        return temp

def getNotes(cur, uuid):
    query="SELECT notes FROM birds_animal WHERE uuid='"+uuid+"'";
    res=cur.execute(query)
    temp=res.fetchall()
    
    if len(temp)>0:
        return temp
    else:
        return 'no notes'

def addToNotes(cur,uuid,note):
    #cur = con.cursor()
    query="SELECT notes FROM birds_animal WHERE uuid='"+uuid+"'";
    res=cur.execute(query)
    temp=res.fetchall()
    bup=str(temp);
    bup=bup+"_fromFMDB_ "+note;
    query=r"UPDATE birds_animal SET notes="+r'"'+bup+r'"'+r" WHERE uuid='"+uuid+r"'";
    res=cur.execute(query)
    
  #  con.commit()   

def getParentsfromUUID(cur,uuid):    
   
    father='NULL'
    mother='NULL'

    query="SELECT parent_id FROM birds_parent WHERE child_id='"+uuid+"'"
    res=cur.execute(query)
    temp=res.fetchall()

    for idx in temp:         
         query="SELECT sex FROM birds_animal WHERE uuid='"+idx[0]+"'"
         res=cur.execute(query)
         parentalsex=res.fetchall()
         try:
             if (idx[0]!='NULL'):
                if parentalsex[0][0]=='M':
                      father=idx
                elif parentalsex[0][0]=='F':
                      mother=idx
         except:
                 pass       
         
    return father, mother

def getGeneticParentsfromUUID(cur,uuid):
    father='NULL'
    mother='NULL'      
    query="SELECT genparent_id FROM birds_geneticparent WHERE genchild_id='"+uuid+"'"
    res=cur.execute(query)

    temp=res.fetchall();    
    for idx in temp:         
         query="SELECT sex FROM birds_animal WHERE uuid='"+idx[0]+"'"
         res=cur.execute(query)
         parentalsex=res.fetchall()
         try:
             if (idx[0]!='NULL'):
                if parentalsex[0][0]=='M':
                      father=idx
                elif parentalsex[0][0]=='F':
                      mother=idx
         except:
                pass

            # pdb.set_trace();
         
    return father, mother


def getMotherfromUUID(cur,uuid):
    father='NULL'
    mother='NULL'      
    query="SELECT genparent_id FROM birds_geneticparent WHERE genchild_id='"+uuid+"'"
    res=cur.execute(query)

    temp=res.fetchall()
    mother=temp[0]
    for idx in temp:         
         query="SELECT sex FROM birds_animal WHERE uuid='"+idx[0]+"'"
         res=cur.execute(query)
         parentalsex=res.fetchall()
      #   pdb.set_trace();
         if parentalsex[0][0]=='M':
             father=idx
         elif parentalsex[0][0]=='F':
             mother=idx
         
    return mother

def getFatherfromUUID(cur,uuid):
    father='NULL'
    mother='NULL'      
    query="SELECT genparent_id FROM birds_geneticparent WHERE sex='M' AND genchild_id='"+uuid+"'";
    res=cur.execute(query)

    temp=res.fetchall();
    mother=temp[0]
    for idx in temp:         
         query="SELECT sex FROM birds_animal WHERE uuid='"+idx[0]+"'";
         res=cur.execute(query)
         parentalsex=res.fetchall();
      #   pdb.set_trace();
         if parentalsex[0][0]=='M':
             father=idx;
         elif parentalsex[0][0]=='F':
             mother=idx;  
         
    return father

def getBirthDate(cur,uuid):
    query="SELECT hatch_date FROM birds_animal WHERE uuid='"+uuid+"'";
    res=cur.execute(query)
    temp=res.fetchall();
    if len(temp) > 0 and temp[0] != '(None,)':
        return temp[0]
    else:
        return ('no match')

def getAllSNPsfrombird(cur, uuid):      
      
    query="SELECT SNP FROM birds_geneticdata WHERE IND_UUID='"+uuid+"'";    
    res=cur.execute(query)
    temp=res.fetchall()
    data=temp[0][0];
    data=data.split(',');
    #if len(temp)>0:
    return data
    #else:
    #    return False

def getSingleLocusSNPsfrombird(cur, uuid, idx):    
          
    query="SELECT gene FROM birds_geneticdata WHERE IND_UUID='"+uuid+"'";    
    res=cur.execute(query)
    temp=res.fetchall()
    data=temp[0][0];
    data=data.split(',');
    
    return data[idx];

def getAllBirdsAtSingleLocusSNPs(cur,idx):
    query="SELECT IND_UUID FROM birds_geneticdata";
    res=cur.execute(query);
    temp=res.fetchall();
    outputSNPS=[0]*len(temp)
    outputUUIDs=[0]*len(temp);
    count=0;
    for uuid in temp:
        
    #    pdb.set_trace();
        query="SELECT SNP FROM birds_geneticdata WHERE IND_UUID='"+uuid[0]+"'";    
        res=cur.execute(query)
        temp=res.fetchall()
        data=temp[0][0];
        data=data.split(',');
        outputUUIDs[count]=uuid[0];
        outputSNPS[count]=data[idx];
        count=count+1;
    
    return outputSNPS,outputUUIDs;
     
def getExpressionDatafromBird(cur,brainregion, uuid):    
    #pdb.set_trace()
    brainregion=brainregion.upper()
    table='bird_'+brainregion+'_RNA'
    birdID=brainregion+"_"+uuid      
    query="SELECT gene,'"+birdID+"' FROM '"+table+"'"    
    res=cur.execute(query)
    temp=res.fetchall()
        
    if len(temp)>0:
        return temp
    else:
        return False

def getAllExpressionDatabyGene(cur,brainregion,gene):
    table='bird_'+brainregion+'_RNA'    
    query="SELECT * from '"+table+"' WHERE gene='"+gene+"'"
    res=cur.execute(query)
    Data=res.fetchall()
    query="SELECT name FROM PRAGMA_TABLE_INFO('"+table+"')"
    res=cur.execute(query)
    Labels=res.fetchall()
    pdb.set_treace()
    
    return Data,Labels

def getAllExpressionData(cur,brainregion):    
    table='bird_'+brainregion+'_RNA'
    query="SELECT * FROM '"+table+"'"
    res=cur.execute(query)
    Data=res.fetchall()
    query="SELECT name FROM PRAGMA_TABLE_INFO('"+table+"')"
    res=cur.execute(query)
    Labels=res.fetchall()    

    return Data,Labels
    
      
def getAllExpressionDataByGene(cur,brainregion,gene):     
    table='bird_'+brainregion+'_RNA'
    query="SELECT * FROM '"+table+"' WHERE gene='"+gene+"'"
    #query="SELECT * from 'bird_RHVC_RNA' where gene='A1CF'" # WHERE gene=*"      
    res=cur.execute(query)
    Data=res.fetchall()
    query="SELECT name FROM PRAGMA_TABLE_INFO('"+table+"')"
    res=cur.execute(query)
    Labels=res.fetchall()    

    return Data[0],Labels

def FulltoAbbrBandNames(birdname):
    birdname=birdname.lower()
    birdname=birdname.replace('pink','pk')
    birdname=birdname.replace('black','bk')
    birdname=birdname.replace('white','wh')
    birdname=birdname.replace('purple','pu')    
    birdname=birdname.replace('blue','bu')
    birdname=birdname.replace('orange','or')
    birdname=birdname.replace('green','gr')    
    birdname=birdname.replace('brown','br')
    birdname=birdname.replace('yellow','ye')
    birdname=birdname.replace('red','rd')

    return birdname
    
    
#con = sqlite3.connect("C:\Data\db.sqlite3");
#cur = con.cursor()
#uuid=getUUIDfromBands(cur,'w41w56')
   
#string=DBBandstoBandStr(cur,155,100,156,5)


#[a]=getBandsfromUUID(cur,(ed2ac2203df64544849a4366f65ba9fc,))
#father,mother=getParentsfromUUID(cur,temp);

#test=addToNotes(con, ('836787cbfa4e44f4bdb19d1977fb4e97',))

    
#data,labels=getAllExpressionData(cur,'rhvc')
#uuid='8eec179e18314cd1a9984e3d5c1f29b3' # in genetic data
#temp=getAllBirdsAtSingleLocusSNPs(cur, 53)
    

#temp=('3f08d0f19c284503a4131138554e2e8b',)
#tempb=('fc840941e62346ef9cc0bb5afb7bf35c',)
#kids=getRearedOffspringfromParents(cur,temp,tempb);
#b=isParent(cur,tempb);

#query="SELECT SNP FROM birds_geneticdata WHERE FATHER_ID='or76bk11'"


#SELECT parent_id FROM birds_parent WHERE child_id = 'e0e879ec16134770a5a9a2c8a3b520d9';
#a23cb510d6564e288f5e87bc1e5905c8=pu33bk14
#05433576064846d09ac14c257765a32=pk54gr20

