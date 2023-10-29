import time

def lecturaPos():
    l_pos = []
    pos_x = 0
    pos_y = 0
    parada = "no"
    while parada != "si":
        pos_x = float(input("Introduce la posición x: "))
        pos_y = float(input("Introduce la posición y: "))
        
        posicion = [pos_x, pos_y]
        l_pos.append(posicion)
        posicion = [0, 0]
        print("Esta es tu lista de posiciones [X,Y]:", l_pos)
        parada = input("Escriba 'si' si deseas parar de recibir datos: ")
    return l_pos
    
def recorridoPos (lista):
	for i in lista:
		print(i)
		
def robotPos (lista):
	for i in lista:
		print("Robot va a la posición", i)
		time.sleep(2)
		lista=lista[1:]
		print("Te quedan las siguientes posiciones:", lista)
		time.sleep(2)
	return lista

lista_pos = []

while True:
    lista_pos = lecturaPos()
    recorridoPos(lista_pos)
    lista_pos = robotPos(lista_pos)
    repetir = input("¿Seguimos más? ")
    if repetir == "no":
    	break
