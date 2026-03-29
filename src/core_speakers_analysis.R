library(tidyverse)
library(entropy)
library(ggplot2)

data_dir <- "."
fig_dir  <- "figures"

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------

df <- readRDS(file.path(data_dir, "meld_utterances_with_sim_emotion.rds"))
speaker_overview <- read_csv(file.path(data_dir, "speaker_overview.csv"),
                             show_col_types = FALSE)
mask_stats <- read_csv(file.path(data_dir, "speaker_masking_stats.csv"),
                       show_col_types = FALSE)

# Core Friends cast
core_speakers <- c("Joey", "Ross", "Rachel", "Phoebe", "Monica", "Chandler")

# Emotion space + valence
sim_emotions <- c(
  "happy", "excited", "content", "neutral",
  "bored", "annoyed", "sad", "frustrated"
)

valence_map <- c(
  happy = 0.8,
  excited = 0.7,
  content = 0.6,
  neutral = 0.0,
  bored = -0.3,
  annoyed = -0.4,
  sad = -0.7,
  frustrated = -0.6
)

# ------------------------------------------------------------
# 2) Per-speaker entropy
# ------------------------------------------------------------

speaker_entropy <- df %>%
  filter(!is.na(sim_emotion)) %>%
  group_by(Speaker) %>%
  summarise(
    emotion_freq = list(table(factor(sim_emotion, levels = sim_emotions))),
    .groups = "drop"
  ) %>%
  mutate(
    probs   = purrr::map(emotion_freq, ~ as.numeric(.x) / sum(.x)),
    entropy = purrr::map_dbl(probs, ~ entropy::entropy(.x, unit = "log2"))
  ) %>%
  select(Speaker, entropy)

# ------------------------------------------------------------
# 3) Per-speaker valence variability
# ------------------------------------------------------------

speaker_variability <- df %>%
  filter(!is.na(sim_emotion)) %>%
  mutate(valence = valence_map[sim_emotion]) %>%
  group_by(Speaker) %>%
  summarise(
    valence_sd   = sd(valence, na.rm = TRUE),
    valence_mean = mean(valence, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 4) Merge everything into one summary table
# ------------------------------------------------------------

final_summary <- speaker_overview %>%
  left_join(mask_stats,        by = "Speaker") %>%
  left_join(speaker_entropy,   by = "Speaker") %>%
  left_join(speaker_variability, by = "Speaker")

write_csv(final_summary, file.path(data_dir, "speaker_full_summary.csv"))

# Filter to the core 6 Friends
fs_core <- final_summary %>%
  filter(Speaker %in% core_speakers)

cat("Core speakers summary:\n")
print(fs_core %>%
        select(Speaker, total_utterances, neg_contexts,
               masking_rate, entropy, valence_sd))

# ------------------------------------------------------------
# 5) Correlations for the core 6
# ------------------------------------------------------------

cor_entropy <- cor(fs_core$masking_rate,
                   fs_core$entropy,
                   use = "complete.obs")

cor_var     <- cor(fs_core$masking_rate,
                   fs_core$valence_sd,
                   use = "complete.obs")

cat("\nCorrelations (core 6):\n")
cat("  r(masking, entropy)     =", cor_entropy, "\n")
cat("  r(masking, valence_sd)  =", cor_var, "\n\n")

# ------------------------------------------------------------
# 6) Plots (core 6)
# ------------------------------------------------------------

p_entropy <- ggplot(fs_core,
                    aes(x = masking_rate, y = entropy)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_text(aes(label = Speaker), vjust = -0.7, size = 3) +
  labs(
    title = "Core Friends: Emotional entropy vs masking rate",
    x = "Masking rate (positivity in negative contexts)",
    y = "Emotion entropy (bits)"
  ) +
  theme_minimal()
p_entropy

p_var <- ggplot(fs_core,
                aes(x = masking_rate, y = valence_sd)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = FALSE) +
  geom_text(aes(label = Speaker), vjust = -0.7, size = 3) +
  labs(
    title = "Core Friends: Valence variability vs masking rate",
    x = "Masking rate (positivity in negative contexts)",
    y = "Valence variability (SD)"
  ) +
  theme_minimal()

p_var

ggsave(file.path(fig_dir, "core_entropy_vs_masking.png"),
       plot = p_entropy, width = 6, height = 4, dpi = 300)

ggsave(file.path(fig_dir, "core_valence_sd_vs_masking.png"),
       plot = p_var, width = 6, height = 4, dpi = 300)

cat("Figures saved in 'figures/' folder.\n")
