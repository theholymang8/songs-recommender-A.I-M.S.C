CREATE TABLE `vector_metadata` (
  `id` int NOT NULL AUTO_INCREMENT,
  `vector_id` int DEFAULT NULL,
  `track_id` int DEFAULT NULL,
  `vector_dimensions` int DEFAULT NULL,
  `faiss_index` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=8192 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci
