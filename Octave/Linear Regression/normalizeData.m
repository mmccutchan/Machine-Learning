function [normalizedData] = normalizeData(data)
  normalizedData = (data - mean(data)) / std(data);
