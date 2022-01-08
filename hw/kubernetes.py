import os

# Question 1
num = 1
ans = 'kind --version'
print(f"Question {num}: {os.system(ans)}")

# Question 2
num = 2
cmd = "sudo kind export kubeconfig"
os.system(cmd)
cmd = "sudo kubectl cluster-info --context kind-kind"
os.system(cmd)
cmd = "sudo kubectl get service"
os.system(cmd)
ans = 'SERVICE IP: 10.96.0.1'
print(f"Question {num}: {ans}")

# Question 3
num = 3
ans = "sudo kind load docker-image churn-model:v001"
print(f"Question {num}: {ans}")

# Question 4
num = 4
ans = '9696'
print(f"Question {num}: {ans}")

# Question 5
num = 5
ans = 'churn'
print(f"Question {num}: {ans}")

# Question 6
num = 6
ans = 'churn-model'
print(f"Question {num}: {ans}")
