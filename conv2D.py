def conv2D(data,kernal):
    n,m=len(data),len(data[0])
    k=len(kernal)
    res=[]
    for i in range(n-k+1):
        row=[]
        for j in range(m-k+1):
            val=0
            for p in range(k):
                for q in range(k):
                    val+=data[i+p][j+q]*kernal[p][q]
            
            row.append(val)
        res.append(row)
    return res

data=[[1, -1, 0],[-3, 0, 2],[8, 9, 1]]
kernal=[[1, -1],[-1, 1]]

result=conv2D(data,kernal)

print(result)