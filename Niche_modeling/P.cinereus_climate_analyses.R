# Plethodon cinereus iNaturalist & computer vision model analyses

library(dplyr)
library(ENMeval)
library(rSDM)
library(dismo)
library(spThin)
library(readr)
library(rmaxent)
library(gtools)
library(scrubr)
library(spatstat)
library(spocc)
library(raster)

#Get point data (cleaned & morphs separated)
setwd("/Users/Maggie/Dropbox/Niche_modeling/Pc_ENM_ML/")
Striped.pts <- read.csv("Striped.pts.csv", header = TRUE, stringsAsFactors = FALSE)
Unstriped.pts <- read.csv("Unstriped.pts2.csv", header = TRUE, stringsAsFactors = FALSE)

#spatial points
xy_data.s <- data.frame(x = Striped.pts$longitude, y = Striped.pts$latitude)
coordinates(xy_data.s) <- ~ x + y
proj4string(xy_data.s) <- crs("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")

xy_data.u <- data.frame(x = Unstriped.pts$longitude, y = Unstriped.pts$latitude)
coordinates(xy_data.u) <- ~ x + y
proj4string(xy_data.u) <- crs("+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84 +towgs84=0,0,0")

#Reduced raster layers 
setwd("/Users/Maggie/Dropbox/Niche_modeling/Reduced_layers_2/")
list <- list.files(full.names = T, recursive = FALSE) 
list <- mixedsort(sort(list))
envtStack <- stack(list)
plot(envtStack)

##########################################

#Looking for more ouliers 
s.cin.pts.scrubbed <- coord_incomplete(Striped.pts) #Removes data with incomplete coordinates
nrow(s.cin.pts.scrubbed) #none
s.cin.pts.scrubbed <- coord_unlikely(s.cin.pts.scrubbed) #Looking for unlikely coordinates 
nrow(s.cin.pts.scrubbed) #none

u.cin.pts.scrubbed <- coord_incomplete(Unstriped.pts) #Removes data with incomplete coordinates
nrow(u.cin.pts.scrubbed) #none
u.cin.pts.scrubbed <- coord_unlikely(u.cin.pts.scrubbed) #Looking for unlikely coordinates 
nrow(u.cin.pts.scrubbed) #none

####
#removing duplicates
#striped
s.cin.data.unique <- s.cin.pts.scrubbed %>%
  distinct
head(s.cin.data.unique) #no overlap

#removing duplicates
u.cin.data.unique <- u.cin.pts.scrubbed %>%
  distinct
head(u.cin.data.unique) #no overlap

#######################################################

#Thin occurrence records
library(spThin)
#25km
striped_thinned_dataset_full_25km <-
  thin( loc.data = s.cin.data.unique, 
        lat.col = "latitude", long.col = "longitude", 
        spec.col = "taxon", 
        thin.par = 25, reps = 100, 
        locs.thinned.list.return = TRUE, 
        write.files = FALSE, 
        write.log.file = FALSE)

summary(striped_thinned_dataset_full_25km)
head(striped_thinned_dataset_full_25km)
km25.s <- as.data.frame(striped_thinned_dataset_full_25km[[100]])
nrow(km25.s)
plot(km25.s)
#write.csv(km25.s, "Striped.pts.25km.csv", row.names = F)

#######################
#Unstriped
#25km
unstriped_thinned_dataset_full_25km <-
  thin( loc.data = u.cin.data.unique, 
        lat.col = "latitude", long.col = "longitude", 
        spec.col = "taxon", 
        thin.par = 25, reps = 100, 
        locs.thinned.list.return = TRUE, 
        write.files = FALSE, 
        write.log.file = FALSE)

summary(unstriped_thinned_dataset_full_25km)
head(unstriped_thinned_dataset_full_25km)
km25.u <- as.data.frame(unstriped_thinned_dataset_full_25km[[100]])
nrow(km25.u)
plot(km25.u)
#write.csv(km25.u, "Unstriped.pts.25km.csv", row.names = F)

############################################################################
############################################################################

#Niche models
#####
#Load data points 

#striped 
setwd("/Users/Maggie/Dropbox/Niche_modeling/")
s.km25.pts <- read.csv("Striped.pts.25km.csv", header = TRUE, stringsAsFactors = FALSE)
str(s.km25.pts)

#Designate background data
bg <- randomPoints(envtStack[[1]], n=10000)
bg <- as.data.frame(bg)
plot(envtStack[[1]], legend=FALSE)
plot(x = bg$x, y = bg$y, pch = 16, col='red')

##Striped model
s.modeval <- ENMevaluate(occ = s.km25.pts[, c("Longitude", "Latitude")], 
                         env = envtStack,
                         bg.coords = bg,
                         method = "block",
                         algorithm = 'maxent.jar',
                         RMvalues = c(0.5, 1, 2, 3, 4, 5),
                         #fc = c("L", "H", "LQ", "LQH", "LQP", "LQPH", "LQPHT"),
                         fc = c("L", "H", "LQH", "LP", "LQ","LQP", "LQPH"), #fc = c("L", "H", "LQH")
                         #overlap = TRUE,
                         #clamp = TRUE, 
                         rasterPreds = TRUE,
                         parallel = TRUE,
                         numCores = 4,
                         bin.output = TRUE,
                         progbar = TRUE)

s.modeval
str(s.modeval, max.level=3)

s.modeval@results

save(s.modeval, file = "striped_ENM_5_28_21.rda")
writeRaster(x = p.s, filename = "striped_ENM_5_28_21.asc", format = "ascii", NAFlag = "-9999", overwrite = T)


#####################################################
##Unstriped model

setwd("/Users/Maggie/Dropbox/Niche_modeling/")
u.km25.pts <- read.csv("Unstriped.pts.25km.csv", header = TRUE, stringsAsFactors = FALSE)
str(u.km25.pts)

u.modeval <- ENMevaluate(occ = u.km25.pts[, c("Longitude", "Latitude")], 
                         env = envtStack,
                         bg.coords = bg,
                         method = "block",
                         algorithm = 'maxent.jar',
                         RMvalues = c(0.5, 1, 2, 3, 4, 5),
                         #fc = c("L", "H", "LQ", "LQH", "LQP", "LQPH", "LQPHT"),
                         fc = c("L", "H", "LQH", "LP", "LQ","LQP", "LQPH"), 
                         #overlap = TRUE,
                         #clamp = TRUE, 
                         rasterPreds = TRUE,
                         parallel = TRUE,
                         numCores = 4,
                         bin.output = TRUE,
                         progbar = TRUE)


u.modeval
str(u.modeval, max.level=3)

u.modeval@results

save(u.modeval, file = "unstriped_ENM_5_28_21.rda")
writeRaster(x = p.u, filename = "unstriped_ENM_5_28_21.asc", format = "ascii", NAFlag = "-9999", overwrite = T)

###############################################################################
###############################################################################
#Niche breadth and overlap

striped_cin <- raster("striped_ENM_5_28_21.asc")
unstriped_cin <- raster("unstriped_ENM_5_28_21.asc")

#niche breadth
s_breadth <- ENMTools::raster.breadth(x = striped_cin)
u_breadth <- ENMTools::raster.breadth(x = unstriped_cin)

#ENM overlap
enm_stack <- stack(striped_cin, unstriped_cin)
names(enm_stack) <- c("striped", "unstriped")

calc.niche.overlap(enm_stack, overlapStat = "D")

########################################################################################################
########################################################################################################
#Best models
load(file = "striped_ENM_5_28_21.rda")
load(file = "unstriped_ENM_5_28_21.rda")

#threshold
# Find the model with the lowest AICc
aicmod.s <- which(s.modeval@results$AICc == min(s.modeval@results$AICc))
aic.opt.s <- s.modeval@models[[which(s.modeval@results$delta.AICc==0)]]
aic.opt.s@results #select threshold 

plot(striped_cin)
rc.s <- reclassify(striped_cin, c(0, 0.1961, 0, 0.1961, 1, 1))
plot(rc.s)

s.pres.pts.n <- rasterToPoints(rc.s)
View(s.pres.pts.n)
#Get presence pts only
pts.sub.s <- subset(as.data.frame(s.pres.pts.n), striped_ENM_5_28_21=="1")
View(pts.sub.s)

#############
#unstriped
#threshold
# Find the model with the lowest AICc
aicmod.u <- which(u.modeval@results$AICc == min(u.modeval@results$AICc))
aic.opt <- u.modeval@models[[which(u.modeval@results$delta.AICc==0)]]
aic.opt@results
plot(unstriped_cin)
rc3 <- reclassify(unstriped_cin, c(0, 0.2163, 0, 0.2163, 1, 1))
plot(rc3)

u.pres.pts.n <- rasterToPoints(rc3)
View(u.pres.pts.n)
is.data.frame(u.pres.pts.n)
u.pres.pts.n <- as.data.frame(u.pres.pts.n)
table(u.pres.pts.n$unstriped_ENM_5_28_21)
pts.sub <- subset(as.data.frame(u.pres.pts.n), unstriped_ENM_5_28_21=="1")
View(pts.sub)

#######################
#extract points from each raster

#bio2
s.bio.pred.val1.bio2 <- raster::extract(bio2, pts.sub.s[,1:2])
u.bio.pred.val1.bio2 <- raster::extract(bio2, pts.sub[,1:2])
#bio5
s.bio.pred.val1.bio5 <- raster::extract(bio5, pts.sub.s[,1:2])
u.bio.pred.val1.bio5 <- raster::extract(bio5, pts.sub[,1:2])
#bio7
s.bio.pred.val1.bio7 <- raster::extract(bio7, pts.sub.s[,1:2])
u.bio.pred.val1.bio7 <- raster::extract(bio7, pts.sub[,1:2])
#bio8
s.bio.pred.val1.bio8 <- raster::extract(bio8, pts.sub.s[,1:2])
u.bio.pred.val1.bio8 <- raster::extract(bio8, pts.sub[,1:2])
#bio9
s.bio.pred.val1.bio9 <- raster::extract(bio9, pts.sub.s[,1:2])
u.bio.pred.val1.bio9 <- raster::extract(bio9, pts.sub[,1:2])
#BIO15 = Precipitation Seasonality
s.bio.pred.val1.bio15 <- raster::extract(bio15, pts.sub.s[,1:2])
u.bio.pred.val1.bio15 <- raster::extract(bio15, pts.sub[,1:2])
#bio18
s.bio.pred.val1.bio18 <- raster::extract(bio18, pts.sub.s[,1:2])
u.bio.pred.val1.bio18 <- raster::extract(bio18, pts.sub[,1:2])
#alt
s.bio.pred.val1.alt <- raster::extract(alt, pts.sub.s[,1:2])
u.bio.pred.val1.alt <- raster::extract(alt, pts.sub[,1:2])

###################################
##PCA 
s.data.val_red <- cbind(s.bio.pred.val1.alt, s.bio.pred.val1.bio2,
                        s.bio.pred.val1.bio5, s.bio.pred.val1.bio7,
                        s.bio.pred.val1.bio8, s.bio.pred.val1.bio9,
                        s.bio.pred.val1.bio15, s.bio.pred.val1.bio18) 

s.data.val_red2 <- as.data.frame(s.data.val_red)
s.data.val_red2$Morph <- "striped"
#rename columns
names(s.data.val_red2)[1:8] <- c("ALT","BIO2","BIO5","BIO7","BIO8","BIO9",
                                 "BIO15","BIO18")

u.data.val_red <- cbind(u.bio.pred.val1.alt, u.bio.pred.val1.bio2,
                        u.bio.pred.val1.bio5, u.bio.pred.val1.bio7,
                        u.bio.pred.val1.bio8, u.bio.pred.val1.bio9,
                        u.bio.pred.val1.bio15, u.bio.pred.val1.bio18) 

u.data.val_red2 <- as.data.frame(u.data.val_red)
u.data.val_red2$Morph <- "unstriped"
#rename columns
names(u.data.val_red2)[1:8] <- c("ALT","BIO2","BIO5","BIO7","BIO8","BIO9",
                                 "BIO15","BIO18")

s.u.bio.data_red <- rbind(s.data.val_red2, u.data.val_red2)
#saveRDS(s.u.bio.data_red, "morph_bio_pts_red.rds")

s.u.bio.data_red <- readRDS("morph_bio_pts_red.rds")
morph.des.red <- s.u.bio.data_red[, 1:8]
morph.sp.red <- s.u.bio.data_red[, 9]
morph.sp.red

#Did on server
#morph.pca.red <- prcomp(morph.des.red, center=T, scale.=T)
morph.pca_red <- readRDS("morph.pca.red.rds")
morph.pca_red
summary(morph.pca_red)

plot(morph.pca_red, type = "l",main="")

# Extract PC axes for plotting
PCAvalues <- data.frame(Morph = s.u.bio.data_red$Morph, morph.pca_red$x)

# Extract loadings of the variables
PCAloadings <- data.frame(Variables = rownames(morph.pca_red$rotation), morph.pca_red$rotation)

# Plot
ggplot(PCAvalues, aes(x = PC1, y = PC2, colour = Morph)) +
  geom_hex(bins = 300, fill = "#000000", alpha = 0) +
  geom_segment(data = PCAloadings, aes(x = 0, y = 0, xend = (PC1*4.5),
                                       yend = (PC2*4.5)), arrow = arrow(length = unit(1/2, "picas")),
               color = "gold", lwd=.8) +
  #geom_point(size = 3) +
  annotate("text", color="gold", cex=4, fontface = "bold", x = (PCAloadings$PC1*6), y = (PCAloadings$PC2*6), 
           label = PCAloadings$Variables) + xlab("PC1 (30.0%)") + ylab("PC2 (26.3%)") +
  theme_minimal() +
  scale_color_manual(values=c("red", "black"))
#scale_fill_manual(values=c("red", "black"))

###############################################################################
##Logistic modeling

s.data.red2 <- as.data.frame(s.data.red)
s.data.red2$Morph <- "Striped"
#rename columns
names(s.data.red2)[1:8] <- c("ALT","BIO2","BIO5","BIO7","BIO8","BIO9",
                             "BIO15","BIO18")

u.data.red2 <- as.data.frame(u.data.red)
u.data.red2$Morph <- "Unstriped"
names(u.data.red2)[1:8] <- c("ALT","BIO2","BIO5","BIO7","BIO8","BIO9",
                             "BIO15","BIO18")

cin_data <- rbind(s.data.red2, u.data.red2)
#write.csv(cin_data, "cin_ENM_data_red.csv", row.names = F)

##########################

cin_data <- read.csv("cin_ENM_data_red.csv", header = TRUE, stringsAsFactors = FALSE)
str(cin_data)
cin_data$Morph2[cin_data$Morph == "Striped"] <- "1"
cin_data$Morph2[cin_data$Morph == "Unstriped"] <- "0"
str(cin_data)
cin_data$Morph2 <- factor(cin_data$Morph2)

##Scale and center continuous predictors 
cin_data <- transform(cin_data, ALT=scale(ALT), BIO2=scale(BIO2), BIO5=scale(BIO5), BIO7=scale(BIO7),
                      BIO8=scale(BIO8), BIO9=scale(BIO9), BIO15=scale(BIO15), BIO18=scale(BIO18))
str(cin_data)

morph_glm1 <- glm(Morph2 ~ ALT + BIO2 + BIO5 + BIO7 + BIO8 + BIO9 + BIO15 + BIO18, family = binomial(), data = cin_data)
summary(morph_glm1)
car::vif(morph_glm1)

library(MuMIn)
options(na.action=na.fail)
morph_glm1.models <- dredge(morph_glm1) #took forever 
morph_glm1.models 

morph_glm2 <- glm(Morph2 ~ ALT + BIO2 + BIO7 + BIO8 + BIO9 + BIO15 + BIO18, family = binomial(), data = cin_data)
summary(morph_glm2)
car::vif(morph_glm2)
library(performance)
r2_nagelkerke(morph_glm2)

#dredge
library(MuMIn)
options(na.action=na.fail)
morph_glm2.models <- dredge(morph_glm2) #took forever 
morph_glm2.models #keep all variables 

library(effects)
#plot(allEffects(morph_glm1))
ALT_plot <- plot(effects::effect("ALT", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="Elevation", rug=FALSE, colors="red")
BIO2_plot <- plot(effects::effect("BIO2", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO2", rug=FALSE, colors="red")
#BIO5_plot <- plot(effects::effect("BIO5", morph_glm1), main=F, ylab="Proportion of striped morphs", xlab="BIO5", rug=FALSE, colors="red")
BIO7_plot <- plot(effects::effect("BIO7", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO7", rug=FALSE, colors="red")
BIO8_plot <- plot(effects::effect("BIO8", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO8", rug=FALSE, colors="red")
BIO9_plot <- plot(effects::effect("BIO9", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO9", rug=FALSE, colors="red")
BIO15_plot <- plot(effects::effect("BIO15", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO15", rug=FALSE, colors="red")
BIO18_plot <- plot(effects::effect("BIO18", morph_glm2), main=F, ylab="Proportion of striped morphs", xlab="BIO18", rug=FALSE, colors="red")

library(gridExtra)
pdf("GLM2_plots_11_9_21.pdf", width = 8, height = 12) 
grid.arrange(ALT_plot, BIO2_plot, BIO7_plot, BIO8_plot, 
             BIO9_plot, BIO15_plot, BIO18_plot, nrow = 4)
dev.off()


