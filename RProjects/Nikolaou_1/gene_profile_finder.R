##################################################################
####    Master in Bioinformatics                              ####
####    BC203 Introduction to R for Bioinformatics            ####
####    Vangelis Theodorakis @ med1p112005@med.uoc.gr         ####
####    Written using extensive googling, material from       ####
####    both the BC203 Introduction to R for Bioinformatics   ####
####    and BIO315 Computational Biology                      ####
####                                                          ####
####    PS. All the major workload was done while crying in   ####
####    despair. The reason is that the assingment was not    ####
####    quite descriptive in terms of steps and methodology   ####
####    so it was more or less a black box for me.             ####
##################################################################

# Read the datasets and the one with the profiles

data<-read.delim("/home/theodor/Master/BC203 Introduction to R for Bioinformatics/BC203_Exercise_general.tsv",header=T,sep="\t")
profiles<-read.delim("/home/theodor/Master/BC203 Introduction to R for Bioinformatics/BC203_Exercise_test.tsv",header=T,sep="\t")

# Concatenate the samples with the profile
data<-cbind(data,Profile2=profiles$Profile2)

# Now that they are concatenated normalize them withe the following formula : x-mean/s^2
dataNormalized<-as.data.frame(scale(data[,2:ncol(data)],center = TRUE, scale = TRUE))

# Categorize my data per column name so later is easier to manipulate
wt_headers<-grep("WT",colnames(data))
tg_headers<-grep("Tg",colnames(data))

x_headers<-grep("X",colnames(data))
y_headers<-grep("Y",colnames(data))
z_headers<-grep("Z",colnames(data))
k_headers<-grep("K",colnames(data))
l_headers<-grep("L",colnames(data))
m_headers<-grep("M",colnames(data))

# Now take the means of the rows in healthy and transgenic samples to generate one column "samples"
wt<-rowMeans(dataNormalized[,wt_headers])
tg<-rowMeans(dataNormalized[,tg_headers])

x<-rowMeans(dataNormalized[,x_headers])
y<-rowMeans(dataNormalized[,y_headers])
z<-rowMeans(dataNormalized[,z_headers])
k<-rowMeans(dataNormalized[,k_headers])
l<-rowMeans(dataNormalized[,l_headers])
m<-rowMeans(dataNormalized[,m_headers])

# Now i want to remove the noise. A good approach is to take their logarithm and see if there are changes in the genes
# between wild type and transgenic samples. I do this taking a statistical significance in regard of the difference.
log2(tg/wt)*10->logFC_Tg
pval_Tg<-vector(mode="numeric", length=length(dataNormalized[,1]))
for(i in 1:length(dataNormalized[,1])){pval_Tg[i]<-t.test(dataNormalized[i,wt_headers], dataNormalized[i,tg_headers])$p.value}
which((logFC_Tg > 1 | logFC_Tg < -1) & pval_Tg<=0.05)->changed_genes

# Hold only the changed gene expressions for the treatments
treatments<-data.frame(x,y,z,k,l,m)[changed_genes,]

# Respectively for my profile
profile<-data.frame(Profile=dataNormalized$Profile2)
profile<-profile[changed_genes,]

# Now merge all the data in a dataframe
cleared_data<-data.frame(x,y,z,k,l,m,Profile=dataNormalized$Profile2)[changed_genes,]

# Correlation test gives the Y as the correct treatment
correlations = cor(treatments,profile)

# PCA and KMeans also gives the Y as the right treatment
pc<-princomp(cleared_data)

fit<-cbind(pc$loadings[,1])

# I want 2 clusters. The one that fits my profile and the others
cluster_test<-kmeans(fit,2,iter.max = 1000)

cluster_test$cluster

plot(fit, col =(cluster_test$cluster +1) , main="K-Means result with 2 clusters")


# According to Hierarchical clustering my profile fits Y ?

distances<-dist(t(cleared_data),method = "euclidean")

clusters<-hclust(distances,method = "centroid")

plot(clusters,hang = -1)


