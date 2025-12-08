library(tidyverse)

# loading the files 
load_meld <- function(path = ".") {
  splits <- c("train", "dev", "test")
  dfs <- lapply(splits, function(s) {
    f <- file.path(path, paste0(s, "_sent_emo.csv"))
    read_csv(f, show_col_types = FALSE) %>%
      mutate(split = s)
  })
  
  bind_rows(dfs) %>%
    select(Dialogue_ID, Utterance_ID, Speaker, Emotion, Sentiment, split) %>%
    arrange(Dialogue_ID, Utterance_ID)
}

df <- load_meld(".")
cat("Total utterances:", nrow(df), "\n")
cat("Number of speakers:", n_distinct(df$Speaker), "\n\n")

# Mapping  MELD emotions -> 8-emotion space (emotions were extracted from the dataset/ the unique filter)

sim_emotions <- c(
  "happy", "excited", "content", "neutral",
  "bored", "annoyed", "sad", "frustrated"
)

meld_to_sim <- c(
  "joy"      = "happy",
  "anger"    = "annoyed",
  "disgust"  = "annoyed",
  "fear"     = "frustrated",
  "sadness"  = "sad",
  "neutral"  = "neutral",
  "surprise" = "excited"
)

df <- df %>%
  mutate(
    Emotion   = tolower(Emotion),
    Sentiment = tolower(Sentiment),
    sim_emotion = recode(Emotion, !!!meld_to_sim, .default = NA_character_)
  ) %>%
  filter(!is.na(sim_emotion))

cat("After mapping, valid mapped utterances:", nrow(df), "\n\n")

# Global transition matrix 

compute_global_transition <- function(df, sim_emotions) {
  df_trans <- df %>%
    group_by(Dialogue_ID) %>%
    arrange(Utterance_ID, .by_group = TRUE) %>%
    mutate(next_emotion = lead(sim_emotion)) %>%
    ungroup() %>%
    filter(!is.na(next_emotion))
  
  tab <- table(df_trans$sim_emotion, df_trans$next_emotion)
  
  full_tab <- matrix(
    0,
    nrow = length(sim_emotions),
    ncol = length(sim_emotions),
    dimnames = list(sim_emotions, sim_emotions)
  )
  full_tab[rownames(tab), colnames(tab)] <- tab
  
  C <- full_tab + 0.1
  P <- C / rowSums(C)
  
  list(counts = full_tab, probs = P)
}

global_trans <- compute_global_transition(df, sim_emotions)
P_global <- global_trans$probs

cat("Global transition matrix shape:",
    paste(dim(P_global), collapse = "x"), "\n\n")

write_csv(
  as.data.frame(P_global) %>%
    mutate(from = rownames(P_global)) %>%
    relocate(from),
  "P_global.csv"
)

# Per-speaker transition matrices 

compute_speaker_transitions <- function(df, sim_emotions,
                                        min_transitions = 30) {
  df_trans <- df %>%
    group_by(Dialogue_ID, Speaker) %>%
    arrange(Utterance_ID, .by_group = TRUE) %>%
    mutate(next_emotion = lead(sim_emotion)) %>%
    ungroup() %>%
    filter(!is.na(next_emotion))
  
  speaker_counts <- df_trans %>%
    count(Speaker, sim_emotion, next_emotion, name = "n")
  
  totals <- speaker_counts %>%
    group_by(Speaker) %>%
    summarise(total = sum(n), .groups = "drop")
  
  valid_speakers <- totals %>%
    filter(total >= min_transitions) %>%
    pull(Speaker)
  
  cat("Speakers with at least", min_transitions,
      "transitions:", length(valid_speakers), "\n\n")
  
  mats <- list()
  
  for (sp in valid_speakers) {
    sub <- speaker_counts %>%
      filter(Speaker == sp)
    
    M <- matrix(
      0,
      nrow = length(sim_emotions),
      ncol = length(sim_emotions),
      dimnames = list(sim_emotions, sim_emotions)
    )
    
    for (i in seq_len(nrow(sub))) {
      from <- sub$sim_emotion[i]
      to   <- sub$next_emotion[i]
      M[from, to] <- M[from, to] + sub$n[i]
    }
    
    C <- M + 0.1
    P <- C / rowSums(C)
    mats[[sp]] <- P
  }
  
  mats
}

speaker_mats <- compute_speaker_transitions(df, sim_emotions,
                                            min_transitions = 30)

cat("Example speaker matrix names (first 5):\n")
print(head(names(speaker_mats), 5))
cat("\n")

saveRDS(speaker_mats, file = "speaker_transition_matrices.rds")

# Masking / people-pleasing proxy 

compute_masking <- function(df, min_contexts = 10) {
  df_ord <- df %>%
    arrange(Dialogue_ID, Utterance_ID)
  
  df_with_prev <- df_ord %>%
    group_by(Dialogue_ID) %>%
    mutate(
      prev_sentiment = lag(Sentiment),
      prev_speaker   = lag(Speaker)
    ) %>%
    ungroup()
  
  neg_context <- df_with_prev %>%
    filter(prev_sentiment == "negative")
  
  mask_stats <- neg_context %>%
    group_by(Speaker) %>%
    summarise(
      neg_contexts     = n(),
      positive_replies = sum(Sentiment == "positive"),
      masking_rate     = positive_replies / neg_contexts,
      .groups = "drop"
    ) %>%
    filter(neg_contexts >= min_contexts) %>%
    arrange(desc(masking_rate))
  
  mask_stats
}

mask_stats <- compute_masking(df, min_contexts = 10)

cat("Speakers with enough negative contexts:", nrow(mask_stats), "\n\n")
cat("Top 5 'pleaser-like' speakers:\n")
print(head(mask_stats, 5))
cat("\n")

write_csv(mask_stats, "speaker_masking_stats.csv")

# Speaker overview 
speaker_overview <- df %>%
  group_by(Speaker) %>%
  summarise(
    total_utterances = n(),
    n_positive       = sum(Sentiment == "positive", na.rm = TRUE),
    n_neutral        = sum(Sentiment == "neutral", na.rm = TRUE),
    n_negative       = sum(Sentiment == "negative", na.rm = TRUE),
    unique_emotions  = n_distinct(sim_emotion),
    .groups = "drop"
  ) %>%
  arrange(desc(total_utterances))

write_csv(speaker_overview, "speaker_overview.csv")
saveRDS(df, "meld_utterances_with_sim_emotion.rds")


