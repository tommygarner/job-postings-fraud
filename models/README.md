# \# Trained Models

# 

# This directory contains the trained machine learning models for the fake job posting detection system.

# 

# \## 📦 Files

# 

# | File | Description | Size | Format |

# |------|-------------|------|--------|

# | `naive\_bayes\_model.pkl` | Naive Bayes classifier trained on TF-IDF + numeric features | ~5-10 MB | Pickle |

# | `lstm\_model.h5` | LSTM neural network for sequential pattern detection | ~50-100 MB | Keras HDF5 |

# | `vectorizer.pkl` | TF-IDF vectorizer (5000 features) | ~20-30 MB | Pickle |

# | `tokenizer.pkl` | Keras tokenizer (vocab\_size=10,000) | ~2-5 MB | Pickle |

# 

# \## 🎯 Model Performance

# 

# \### Final Ensemble (Weighted 25/75 NB/LSTM + Rule-Boosting)

# 

# | Metric | Value | Description |

# |--------|-------|-------------|

# | \*\*Accuracy\*\* | 97.52% | Overall correct predictions |

# | \*\*F1-Score\*\* | 0.7140 | Harmonic mean of precision and recall |

# | \*\*Precision\*\* | 80.98% | When flagged as fraud, correct 81% of time |

# | \*\*Recall\*\* | 63.85% | Directly catches 64% of frauds |

# | \*\*Total Detection\*\* | 76.9% | Frauds flagged as HIGH or MEDIUM risk |

# 

# \### Risk-Based Performance

# 

# | Risk Level | Fraud Detection | False Positive Rate |

# |-----------|-----------------|---------------------|

# | 🔴 \*\*HIGH RISK\*\* | 59.2% (154/260) | 0.3% (17/5104) |

# | 🟡 \*\*MEDIUM RISK\*\* | 17.7% (46/260) | 4.5% (229/5104) |

# | 🟢 \*\*LOW RISK\*\* | 23.1% (60/260) | 95.2% (4858/5104) |

# 

# \### Individual Model Performance

# 

# | Model | Mean Score (Fraud) | Mean Score (Real) | Strength |

# |-------|-------------------|-------------------|----------|

# | \*\*Naive Bayes\*\* | 31.76% | 1.49% | Fast, interpretable TF-IDF features |

# | \*\*LSTM\*\* | 60.43% | 2.82% | Sequential pattern detection |

# 

# \## 🔧 Model Architecture

# 

# \### Naive Bayes Classifier



