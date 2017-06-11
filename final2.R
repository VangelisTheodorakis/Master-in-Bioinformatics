#exei gami8ei to simpan
#den mporoume na paroume ton xaraktirismo twn metrisewn (cancer - no cancer)
#opote prepei na ta broume apo allou
#auto to kanw apo to arxeiaki SraRunTable
#alla den einai me tin idia seira kai den exoun to idio mege8os
#3ekiniste na bazete tis biblio8ikes toulaxiston giati emena se linux gamiotan kai ekei to simpan

#Define a function for loading packages

load_ = function(pkg, bioC=T) {
	#character.only has to be set to True in order for require() or library() to realize it's dealing with a variable
	if(!require(pkg, character.only=T, quietly = T)) {
		if(bioC){
			source(file = "http://bioconductor.org/biocLite.R")
			biocLite(package, dependencies=T)
		} else {
			install.packages(pkg)
		}
	}
	library(pkg, character.only=T)
}

load_("codetools", bioC=F)
load_("doRNG", bioC=F)
load_("recount")
load_("DESeq2")
load_("edgeR")

#epeidi o lagani gamietai kai prepei na paroume ta xaraktiristika apo allou
my_data<-read.table("/home/antonios/Dropbox/code_base/203/SraRunTable.txt",header = TRUE,sep="\t")

#project of interest
project_id <- 'SRP018008';

# Download the gene-level RangedSummarizedExperiment data
download_study(project_id)

# Load the object rse_gene
load(file.path(project_id, 'rse_gene.Rdata'))
class(rse_gene)

#extracting the count data
count_data <- assay(rse_gene)
count_data[1:5, 1:5]

#length(my_data$SRA_Sample_s)
#length(rse_gene$sample)

characteristics_vec<-c()
for (i in 1:length(rse_gene$sample)){
	name<-rse_gene$sample[i]
	for (j in 1:length(my_data$SRA_Sample_s)){
		name2<-my_data$SRA_Sample_s[j]
		if(name==name2){
			characteristics_vec[i]<-my_data$Sample_Name_s[j]
			break
		}
	}
}

characteristics_vec2<-c()
for(i in 1:length(characteristics_vec)){
	characteristics_vec2[i]<-as.character(my_data$Sample_Name_s[characteristics_vec[i]])
}
characteristics_vec2

for(i in 1:length(characteristics_vec2)){
	if(grepl("Cancer",characteristics_vec2[i])){
		characteristics_vec2[i]<-"Cancer"
	}
	else{
		characteristics_vec2[i]<-"Normal"
	}
}
characteristics_vec2

toKeep <- apply(count_data, 1, sum) > 50 * dim(count_data)[2];
count_data <- count_data[toKeep, ];
dim(count_data)


#my_dataset<-count_data
#my_dataset <- rbind(my_dataset,'condition'=characteristics_vec2)

#object for edgeR
dgList <- DGEList(counts=count_data, genes=rownames(count_data))

#design matrix
designMatrix <- model.matrix(~ characteristics_vec2 )

#estimating dispersion
dgList <- estimateGLMCommonDisp(dgList, design=designMatrix)
edgeRFit <- glmFit(dgList, designMatrix)

#testing for disease.status
diseaseGenes <- glmLRT(edgeRFit, coef = 1)

#top differentially expressed genes
diseaseGenes <- topTags(diseaseGenes, n = dim(count_data)[1], p.value = 1)@.Data[[1]]
head(diseaseGenes)

#grammi 67 me grammi 76 mallon den xreiazontai giati to characteristics_vec2 periexei to phenotype
# Extract the sample characteristics
#sample_info <- colData(rse_gene)
#colnames(sample_info)
#sample_info$characteristics[1]

#some more work to to for the actual pheno information
#geochar <- lapply(split(sample_info, seq_len(nrow(sample_info))), geo_characteristics)
#sample_info <- do.call(rbind, characteristics_vec2)
#head(sample_info)

#extracting the information about the genes
#gene_info <- rowData(rse_gene)
#gene_info[1:5, ]
