library(dplyr)
library(sf)
library(ggplot2)
library(sp)

# for england
country = "england"



setwd(paste0("/data/output/",country))

res = read.csv("en_map_plot_5.csv")
res$X=NULL


shapes_in = readRDS("/data/gadm1_nuts3_counties/geometries/gadm1_nuts3_counties_sf_format.Rds")

shape_names = read.csv("data/shape_names_fixed.csv")

shapes_in$name = shape_names 
shapes_it = shapes_in[shapes_in$country=="GBR",]
#shapes_it$key = NULL
shapes_it$country = NULL
codes = read.csv("new_br.csv")
codes =codes[codes$code!="-",]




#------- keep only the nodes in shapes
tmp = data.frame(shapes_it)[,c("key","name")]
tmp$tmp = tmp$name[[1]]
tmp$name = NULL

tmp = merge(tmp,codes,"x","tmp")


tmp = merge(tmp,res,by.x="code",by.y="name")

lp = tmp %>% group_by(key) %>% summarise(rel =mean(relative),real =sum(real),
                                         avg_cases =mean(avg_cases),
                                         cases =mean(cases)
)
lp = data.frame(lp)


dat_map <- 
  left_join(lp,#right_join
            shapes_it,
            by=c("key"="key")) %>% 
  st_as_sf


#------- plot relative
dat_map <- dat_map %>% 
  mutate(rel_discrete = case_when(
    rel < 0.2 ~ "<20%",
    rel < 0.3 ~ "<30%",
    rel < 0.4 ~ "<40%",
    rel < 0.5 ~ "<50%",
    rel < 0.6 ~ "<60%",
    rel < 0.8 ~ "<80%",
    rel < 1 ~ "<100%",
    rel > 1 ~ ">100%")) %>% 
  mutate(rel_discrete = factor(rel_discrete, 
                               levels=c("<20%", "<30%", "<40%", "<50%","<60%","<80%","<100%", ">100%")))



ggplot(st_transform(dat_map, "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")) +
  geom_sf(aes(fill = rel_discrete), colour="#ADADAD", lwd=0) +
  #geom_sf(data=curr_region_outline, fill="#A00000", colour="#A00000", size=1) +
  labs(fill = "Relative\nError") +
  theme_void() +
  scale_fill_brewer(palette = "Reds", na.value="#F5F5F5", drop=FALSE) +
  theme(legend.title = element_text(size = 30), 
        legend.text  = element_text(size = 30),
        legend.key.size = unit(0.8, "lines"), panel.border = element_blank(),
        legend.position = "right", legend.box = "vertical") 

ggsave(paste0("/figures/",country,"_relative_5.pdf"))

#------- plot avg cases
x1 <- ceiling(quantile(dat_map$avg_cases, .3))
x2 <- ceiling(quantile(dat_map$avg_cases, .5))
x3 <- ceiling(quantile(dat_map$avg_cases, .7))
x4 <- ceiling(quantile(dat_map$avg_cases, .80))
x5 <- ceiling(quantile(dat_map$avg_cases, .88))
x6 <- ceiling(quantile(dat_map$avg_cases, .93))
x7 <- ceiling(quantile(dat_map$avg_cases, .96))
x8 <- ceiling(quantile(dat_map$avg_cases, .98))

dat_map <- dat_map %>% 
  mutate(avg_discrete = case_when(
    avg_cases <  x1~ paste0("<",x1),
    avg_cases <  x2~ paste0("<",x2),
    avg_cases <  x3~ paste0("<",x3),
    avg_cases <  x4~ paste0("<",x4),
    avg_cases <  x5~ paste0("<",x5),
    avg_cases <  x6~ paste0("<",x6),
    avg_cases <  x7~ paste0("<",x7),
    avg_cases <  x8~ paste0("<",x8),
    avg_cases >  x8~ paste0(">",x8)#,
    #avg_cases > x9 ~ paste0(">",x9)
  )) %>% 
  mutate(avg_discrete = factor(avg_discrete, levels=c(paste0("<",x1), paste0("<",x2), 
                                                      paste0("<",x3), paste0("<",x4),paste0("<",x5),
                                                      paste0("<",x6),paste0("<",x7),paste0("<",x8),
                                                      paste0(">",x8))))


ggplot(st_transform(dat_map, "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")) +
  geom_sf(aes(fill = avg_discrete), colour="#ADADAD", lwd=0) +
  #geom_sf(data=curr_region_outline, fill="#A00000", colour="#A00000", size=1) +
  labs(fill = "Avg Case\nPer Day") +
  theme_void() +
  scale_fill_brewer(palette = "Blues", na.value="#F5F5F5", drop=FALSE) +
  theme(legend.title = element_text(size = 30), 
        legend.text  = element_text(size = 30),
        legend.key.size = unit(0.8, "lines"),panel.border = element_blank(),
        legend.position = "right", legend.box = "vertical")# +
#guides(fill = guide_legend(nrow = 1, title.hjust = 0.5))


ggsave(paste0("/figures/",country,"_cases.pdf"))
