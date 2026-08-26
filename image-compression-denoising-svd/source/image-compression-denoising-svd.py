import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#PART 2: Preparing the image 

#Composition of Matrix A
matrix_A = plt.imread('070240413_ReyyanBeyzaAydin_grayscalephoto.png')
# We make sure that it meets the requirements of 512 => m,n => 256
rows, columns = matrix_A.shape
print(f" Image has m= {rows} rows and n= {columns} columns meeting the criteria of of 512 => m,n => 256")
r_val=min(rows,columns)
matrix_A= matrix_A.astype(float)
#Composition of Transpose of Matrix A
matrix_A_T= np.transpose(matrix_A)

#Matrix multiplication of A and A^T: B = A^TA
B= matrix_A_T @ matrix_A 
ranks= [5,10,20,50,100,200]
compression_csv_data=[]
frobenius_list= []

#PART 3: Creating an Eigensolver using Power Iteration with Deflation
np.random.seed(0)
eigenvalues =[]
eigenvectors= []
singular_vals= []
B_deflated = B.copy()
num_iterations= 500 # inner loop is to find a single eigenvector

for j in range(max(ranks)): 
     #We generate a random vector to initialize power iteration and normalize the generated vector
    b= np.random.rand(columns)
    b= b/ np.linalg.norm(b) 

    for i in range(num_iterations):
        # we update the b after multiplying eith square matrix B 
        b_updated = np.dot(B_deflated,b)                       
        # then we find the norm of the updated b 
        b_updated_norm= np.linalg.norm(b_updated)
        # finally divide updated b with its norm to acquire the new b
        b_new = b_updated / b_updated_norm

        if np.allclose(b, b_new):
            b= b_new
            print(f"Converged at {i}")
            break
        b = b_new


    eigenvalues.append(b_updated_norm)
    eigenvectors.append(b)


     # Deflation Part: subtracting current component from matrix
    current_ev = np.dot(b.T, np.dot(B_deflated,b))
    B_deflated= B_deflated- current_ev*np.outer(b,b)
        
singular_vals= np.sqrt(eigenvalues)  

#PART 4: Implementing SVD

for k_rank in ranks:
    print(f"Preparing image for rank k= {k_rank}...")

    # BUILDING MATRIX V^T
    matrix_V_T=[]
    for i in range(k_rank):
        matrix_V_T.append( eigenvectors[i]/ np.linalg.norm(eigenvectors[i]))
    matrix_V_T= np.array(matrix_V_T)

    # BUILDING MATRIX U
    matrix_U=np.zeros((rows, k_rank))
    for i in range(k_rank):
        if singular_vals[i]> 1e-9:
            matrix_U[:,i]=(matrix_A @ (eigenvectors[i]/ np.linalg.norm(eigenvectors[i])))/singular_vals[i]
        else:
            matrix_U[:,i]=0

    # BUILDING MATRIX Σ
    matrix_Σ= np.zeros((k_rank,k_rank))
    for i in range(k_rank):
        if i < min(rows, columns):
            matrix_Σ[i,i]= singular_vals[i]
            
    #Bringing the three matricies together: A= UΣV^T
    new_A= matrix_U @ matrix_Σ @ matrix_V_T
    new_A=np.clip(new_A, 0.0, 1.0)

    # PART 6:Running experiments, and collecting data 

    # Calculation of graph variables 
    #Mean Squared Error Calculation
    MSE= np.mean((matrix_A-new_A)**2)

    #Relative Frobenius Error Calculation
    difference_FE= np.linalg.norm(matrix_A-new_A, ord='fro')
    A_FE= np.linalg.norm(matrix_A, ord='fro')
    Frobenius_Err= difference_FE / A_FE
    frobenius_list.append(Frobenius_Err)

    if k_rank<= 0.05* r_val:
        quality_level = "low"
    elif 0.05 * r_val < k_rank <= 0.20 *r_val:
        quality_level = "medium"
    else:
        quality_level = "high"

    compression_csv_data.append({"k": k_rank, "MSE": MSE, "Frobenius": Frobenius_Err, "quality level": quality_level})
    
    #Image based on the input k rank
    plt.figure()
    plt.imshow(new_A, cmap='gray')
    plt.axis('off')
    plt.title(f"Compressed Image Reconstruction (k = {k_rank})")
    plt.savefig(f"compressed_k_{k_rank}.png")
    plt.close()

df_compression = pd.DataFrame(compression_csv_data)
df_compression.to_csv("results.csv", index=False)
print("'results.csv' is saved successfully!")

# Calculation of storage compression  
initial_storage= rows * columns
ratio=[]

for k in ranks:
    compressed_stor= k*(columns+rows+1)
    current_ratio= compressed_stor/initial_storage
    ratio.append(current_ratio)

# The plotting of Relative Frobenius Error Calculation vs. K Rank Values

plt.figure()
plt.plot(ranks, frobenius_list, marker='o', color='purple')
plt.title("Relative Frobenius Error Calculation vs. K Values")
plt.ylabel('Frobenius Error')
plt.xlabel('K Rank Values')
plt.savefig(f"Relative_Frobenius_Error_Calculation_vs._K_Values.png")
plt.close()


# The plotting of Storage Compression Estimate vs. K Rank Values
plt.figure()
plt.plot(ranks, ratio, marker='o', color='green')
plt.title("Storage Compression Estimate vs. K Rank Values")
plt.ylabel('Storage Compression Estimate')
plt.xlabel('K Rank Values')
plt.savefig(f"Storage_Compression_Estimate_vs_K_Rank_Values.png")
plt.close()


#PART 7:Image denoising with truncated SVD + PSNR

# Generating the Noise and Running Noisy SVD

sigma= 15.0 / 255.0
np.random.seed(42)
noise = np.random.normal(0,sigma,(rows, columns))
A_noisy = np.clip(matrix_A + noise, 0.0, 1.0)

plt.figure()
plt.imshow(A_noisy,cmap='gray')
plt.axis('off')
plt.title("Noisy Image")
plt.savefig("Noisy_image.png")
plt.close()

# This functions calculates PSNR for normalized images that has range 0-1, which is our case
def calculate_psnr( clean_img, altered_img):

    psnr_mse= np.mean((clean_img-altered_img)**2)

    if psnr_mse== 0: ## if images are identical and the difference is zero, we don't neeed to conitnue calculating PSNR value since it will be 0
        return psnr_mse, float('inf')
    
    max_pixel= 1.0

    psnr_val= 10 * np.log10((max_pixel ** 2)/ psnr_mse)

    return psnr_mse, psnr_val

mse_noisy, psnr_noisy= calculate_psnr(matrix_A, A_noisy)
print(f"Initial Noisy Image PSNR vs Clean Image: {psnr_noisy} dB")
denoise_csv_data= []
psnr_list = []

B_noisy= np.transpose(A_noisy) @ A_noisy

for k_rank in ranks:

    print(f"Denoising the image using rank k = {k_rank}...")

    B_deflated_n= B_noisy.copy()

    # Now we run power iteration on the noisy matrix. We repeat the same procedure with PART 3.

    eigenvalues_n= []
    eigenvectors_n= []

    for j in range (k_rank):
        b = np.random.rand(columns)
        b= b/ np.linalg.norm(b)

        for i in range(num_iterations):
            b_updated = np.dot( B_deflated_n,b)
            b_updated_norm= np.linalg.norm(b_updated)
            b_new= b_updated / b_updated_norm

            if np.allclose(b, b_new):
                b= b_new
                break
            b = b_new

        eigenvalues_n.append(b_updated_norm)
        eigenvectors_n.append(b)

        # Deflating the noisy matrix

        current_ev= np.dot(b.T, np.dot(B_deflated_n, b))
        B_deflated_n = B_deflated_n- current_ev* np.outer( b, b)
    
    singular_vals_n = np.sqrt(eigenvalues_n)

    # Now we contruct the denoised image matrix following the steps from Part 4

    # Building Matrix V^T
    matrix_V_T_n = []

    for i in range(k_rank):
        v_norm = eigenvectors_n[i] / np.linalg.norm(eigenvectors_n[i])
        matrix_V_T_n.append(v_norm)
    matrix_V_T_n = np.array(matrix_V_T_n)

    # Building Matrix U
    matrix_U_n=np.zeros((rows, k_rank))
    for i in range(k_rank):
        if singular_vals_n[i] > 1e-9:
            v_norm = eigenvectors_n[i] / np.linalg.norm(eigenvectors_n[i])
            matrix_U_n[:,i]=(A_noisy @ v_norm)/singular_vals_n[i]
        else:
            matrix_U_n[:,i]=0

    # Building Matrix Σ
    matrix_Σ_n= np.zeros((k_rank,k_rank))
    for i in range(k_rank):
        if i < min(rows, columns):
            matrix_Σ_n[i,i]= singular_vals_n[i]

    # Bringing the three matricies together: A_hat= UΣV^T 

    A_hat_k = matrix_U_n @ matrix_Σ_n @ matrix_V_T_n
    A_hat_k = np.clip(A_hat_k, 0.0, 1.0) #We make sure that the pixel is in the range

    mse_denoised, psnr_denoised = calculate_psnr(matrix_A, A_hat_k)
    psnr_list.append(psnr_denoised)

    # The dictionary to be used in deoine csv data file

    row_data = {
        "sigma" : sigma,
        "k" : k_rank,
        "MSE noisy" : mse_noisy,
        "PSNR noisy": psnr_noisy,
        "MSE denoised" : mse_denoised,
        "PSNR denoised": psnr_denoised,}
    denoise_csv_data.append(row_data)

    plt.figure()
    plt.imshow(A_hat_k, cmap='gray')
    plt.axis('off')
    plt.title(f"Denoised Image Reconstruction (k= {k_rank})")
    plt.savefig(f"denoised_k_{k_rank}.png")
    plt.close()

# Finding the optimal rank k that mazimizes the filtered image PSNR
best_index= 0
highest_psnr = psnr_list[0]

# The lookup loop is to find maximum value
for index in range (len(psnr_list)):
    if psnr_list[index] > highest_psnr:
        highest_psnr = psnr_list[index]
        best_index = index

# We add the optimal k value to dictionary rows for csv
k_optimal = ranks[best_index]
print(f"The optimal rank k thst maximizes PSNR is: {k_optimal}")

for row in denoise_csv_data:
    row["k optimal"] = k_optimal

# We convert the dataset to a Pandas DataFrame and save as a csv file
df_denoise = pd.DataFrame(denoise_csv_data)
df_denoise.to_csv("denoise_results.csv", index= False)
print("'denoise_results.csv' has been generates and saced successfully!")

# Final Plotting: PSNR vs. Rank K Curve

plt.figure()
plt.plot(ranks, psnr_list, marker ='o', color = 'red', label = 'Denoised PSNR')
plt.title(" Denoising Performance: PSNR vs. K Rank")
plt.xlabel("Rank K")
plt.ylabel("PSNR (dB)")
plt.legend()
plt.savefig(f"Denoising_Performance_PSNR_vs_K_Rank.png")
plt.close()