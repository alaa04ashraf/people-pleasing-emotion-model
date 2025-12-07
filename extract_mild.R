library(tidyverse)


data_dir <- "data"

if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

# The Emotion space I used across R + Python
sim_emotions <- c(
  "happy", "excited", "content", "neutral",
  "bored", "annoyed", "sad", "frustrated"
)

# Mapping from MELD emotion labels to our 8-emotion space - unique/consistent with the dataset
meld_to_sim <- c(
  joy      = "happy",
  anger    = "annoyed",
  disgust  = "annoyed",
  fear     = "frustrated",
  sadness  = "sad",
  neutral  = "neutral",
  surprise = "excited"
)

valence_map <- c(
  happy = 0.8, excited = 0.7, content = 0.6, neutral = 0.0,
  bored = -0.3, annoyed = -0.4, sad = -0.7, frustrated = -0.6
)

# ------------------------------------------------------------
# 2. Load CSVs 
# ------------------------------------------------------------

load_meld <- function(path = "data") {
  splits <- c("train", "dev", "test")
  dfs <- lapply(splits, function(s) {
    f <- file.path(path, paste0(s, "_sent_emo.csv"))
    read_csv(f, show_col_types = FALSE) %>%
      mutate(split = s)
  })
  df_all <- bind_rows(dfs) %>%
    select(Dialogue_ID, Utterance_ID,
           Speaker, Emotion, Sentiment, split) %>%
    arrange(Dialogue_ID, Utterance_ID)
  df_all
}

df <- load_meld(data_dir)

df <- df %>%
  mutate(
    Emotion   = tolower(Emotion),
    Sentiment = tolower(Sentiment),
    sim_emotion = recode(Emotion, !!!meld_to_sim, .default = NA_character_)
  ) %>%
  filter(!is.na(sim_emotion))

# in case 
saveRDS(df, file.path(data_dir, "meld_utterances_with_sim_emotion.rds"))

# ------------------------------------------------------------
# 3. Speaker overview stats
# ------------------------------------------------------------

speaker_overview <- df %>%
  group_by(Speaker) %>%
  summarise(
    total_utterances = n(),
    n_positive       = sum(Sentiment == "positive", na.rm = TRUE),
    n_neutral        = sum(Sentiment == "neutral", na.rm = TRUE),
    n_negative       = sum(Sentiment == "negative", na.rm = TRUE),
    unique_emotions  = n_distinct(sim_emotion),
    mean_valence     = mean(valence_map[sim_emotion], na.rm = TRUE),
    sd_valence       = sd(valence_map[sim_emotion], na.rm = TRUE),
    .groups = "drop"
  )

write_csv(speaker_overview, file.path(data_dir, "speaker_overview.csv"))

# ------------------------------------------------------------
# 4. Masking stats (positivity in negative contexts)
# ------------------------------------------------------------

mask_stats <- df %>%
  arrange(Dialogue_ID, Utterance_ID) %>%
  group_by(Dialogue_ID) %>%
  mutate(prev_sentiment = lag(Sentiment)) %>%
  ungroup() %>%
  filter(prev_sentiment == "negative") %>%
  group_by(Speaker) %>%
  summarise(
    neg_contexts     = n(),
    positive_replies = sum(Sentiment == "positive", na.rm = TRUE),
    masking_rate     = positive_replies / neg_contexts,
    .groups = "drop"
  ) %>%
  filter(neg_contexts >= 10)

write_csv(mask_stats, file.path(data_dir, "speaker_masking_stats.csv"))

# ------------------------------------------------------------
# 5. Transition matrices (per speaker + global)
# ------------------------------------------------------------

# Per-speaker next-emotion transitions (within dialogue & speaker)
df_trans <- df %>%
  group_by(Dialogue_ID, Speaker) %>%
  arrange(Utterance_ID, .by_group = TRUE) %>%
  mutate(next_emotion = lead(sim_emotion)) %>%
  ungroup() %>%
  filter(!is.na(next_emotion))

speakers <- unique(df_trans$Speaker)

speaker_mats <- list()
speaker_transition_long <- list()

for (sp in speakers) {
  sub <- df_trans %>% filter(Speaker == sp)
  M <- matrix(0, nrow = length(sim_emotions), ncol = length(sim_emotions),
              dimnames = list(sim_emotions, sim_emotions))
  
  tab <- table(sub$sim_emotion, sub$next_emotion)
  M[rownames(tab), colnames(tab)] <- tab
  
  # small smoothing and normalizing rows
  M <- M + 0.1
  M <- M / rowSums(M)
  
  speaker_mats[[sp]] <- M  
  long_df <- as.data.frame(as.table(M))
  colnames(long_df) <- c("from", "to", "prob")
  long_df$Speaker <- sp
  speaker_transition_long[[sp]] <- long_df
}

saveRDS(speaker_mats, file.path(data_dir, "speaker_transition_matrices.rds"))

speaker_transition_long_df <- bind_rows(speaker_transition_long)
write_csv(speaker_transition_long_df,
          file.path(data_dir, "speaker_transition_long.csv"))

# Global transition matrix (ignoring speaker)
M_global <- matrix(0, nrow = length(sim_emotions), ncol = length(sim_emotions),
                   dimnames = list(sim_emotions, sim_emotions))
tab_global <- table(df_trans$sim_emotion, df_trans$next_emotion)
M_global[rownames(tab_global), colnames(tab_global)] <- tab_global

M_global <- M_global + 0.1
M_global <- M_global / rowSums(M_global)

P_global <- as.data.frame(M_global)
P_global$from <- rownames(M_global)
P_global <- P_global %>%
  relocate(from)

write_csv(P_global, file.path(data_dir, "P_global.csv"))
