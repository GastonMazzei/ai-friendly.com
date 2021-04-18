import pandas as pd
from os import listdir

MAX_INCARD_LENGTH = 30000
MAX_LEARNCARD_LENGTH = 30000
MAX_COLUMN_LENGTH = 50

uploadedFilesFilenameSize = 7

def validateLearnCardFile(fileName):
	result = {'Nerrors': 0, 'errors': [], 'card': None}
	
	print(fileName)
	try:
		learnCard = pd.read_excel(fileName, header=0)
		result['card'] = learnCard
	except:
		try:
			None #checkeamos .csv o al pedo?
		except:
			None
			
		result['Nerrors'] += 1
		result['errors'].append('Error: Invalid or missing LearnCard!')
		return result
	
	if (learnCard.shape[0] > MAX_LEARNCARD_LENGTH): #Learn row size
		result['Nerrors'] += 1
		result['errors'].append(f"Error: No more than {MAX_LEARNCARD_LENGTH} rows are accepted in LearnCard due to CPU limitations")
		
	if (learnCard.shape[1] > MAX_COLUMN_LENGTH): #Learn column size
		result['Nerrors'] += 1
		result['errors'].append(f"Error: No more than {MAX_COLUMN_LENGTH} columns are accepted in LearnCard due to CPU limitations")
	
	return result


def validateInCardFile(fileName, learnCardFile):
	#podriamos checkear que las columnas sean las mismas tambien
	result = validateLearnCardFile(learnCardFile)
	
	
	print('======== InCardValidation ========')
	print('In    :: %s'%fileName)
	print('Learn :: %s'%learnCardFile)
	try:
		inCard = pd.read_excel(fileName, header=0)
		result['card'] = inCard
	except:
		result['Nerrors'] += 1
		result['errors'].append('Error: Invalid or missing InCard!')
		return result
	
	if result['Nerrors'] == 0:
		learnCard = pd.read_excel(learnCardFile, header=0)
		if inCard.shape[1] != learnCard.shape[1]-1 : #InCard column size
			result['Nerrors'] += 1
			result['errors'].append("Error: The number of columns in InCard must be one less than the number of columns in LearnCard.")
		if inCard.shape[0] > MAX_INCARD_LENGTH:
			result['Nerrors'] += 1
			result['errors'].append(f"Error: No more than {MAX_INCARD_LENGTH} columns are accepted in InCard fue to CPU limitations.")
	
	return result
	 
	
def validateEmail(direccion):
	end_validos = [".edu.ar", ".gob.ar", ".gov.ar", "cedie.org.ar", "uba.ar","jpmorgan.com","gmail.com",
                       "outlook.com", "hotmail.com", "df.com", "yahoo.com", "hotmail.com.ar", "yahoo.com.ar"]
	otros = ["unsam", "sarasa"]
	
	if any(  [direccion[-len(x):]==x for x in end_validos]  ): return True
	if any(  [x in direccion for x in otros]  ): return True

def LearnCardFile(directorio):
	lista = [archivo for archivo in listdir(directorio) if archivo[:3] == 'LC_']
	if len(lista) == 1: return lista[0]
	else: raise Exception('Archivo removido!')

def InCardsFiles(directorio):
	print('DIRECTORIO :: %s'%directorio)
	lista = [archivo for archivo in listdir(directorio) if archivo[:3] == 'IC_']
	if len(lista) >= 1: return sorted(lista)
	else: raise Exception('Archivo(s) removido(s)!')

def nCardFile(fileName):
	return int(fileName[3:6])

def OutCardsFiles(directorio):
	lista = [archivo for archivo in listdir(directorio) if archivo[:3] == 'OC_']
	if len(lista) >= 1: return sorted(lista)
	else: raise Exception('Archivo(s) removido(s)!')
	
#dire = 'uploads/21a9acf9-34ff-4689-b6a5-9262e54ba0f4'
#validateEmail("")
#res = validateLearnCardFile(dire + '/LC_000_LearnCard_Islands.xlsx')
#print(res)
#res1 = validateInCardFile("~/Téléchargements/InCard_House.xlsx", res['card'])
#print(res['Nerrors'])
#print(res['errors'])
#print(res['card'])
