def knn(x_train,y_train,x_new,k):
    distance=[]

    for i in range(len(x_train)):
        dist=0
        for j in range(len(x_train[0])):
            dist+=(x_train[i][j]-x_new[j])**2
        distance.append((dist**0.5,y_train[i]))

    distance.sort()

    k_near=distance[:k]

    label={}

    for _,l in k_near:
        if l is not label:
            label[l]=0
        label[l]+=1
    # Step 5: Pick label with max count
    predicted_label = None
    max_count = -1
    for label, count in label.items():
        if count > max_count:
            max_count = count
            predicted_label = label

    return predicted_label

x_train=[[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]]
y_train=[0, 0, 0, 1, 1, 1]
x_new=[2,2]
k=4

# should return 0
output=knn(x_train,y_train,x_new,k)
print(output)