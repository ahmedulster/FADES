# FADES: Federated Authentication with Adaptive Differential Privacy, Encryption, and Zero-Knowledge Proofs for Secure IoMT Devices

## Abstract

In the rapidly evolving landscape of healthcare technology, the Internet of Medical Things (IoMT) has revolutionised patient care by enabling real-time monitoring and data-driven diagnostics through interconnected medical devices. However, this technological advancement brings significant privacy and security challenges that could potentially compromise sensitive patient information. Our research addresses these critical concerns by introducing FADES—a comprehensive security framework specifically designed for federated learning in resource-constrained IoMT environments.

FADES represents a pioneering approach that integrates four sophisticated security mechanisms: Adaptive Differential Privacy, Lightweight Homomorphic Encryption, Zero-Knowledge Proofs, and Federated Authentication. Through rigorous testing on the WESAD (Wearable Stress and Affect Detection) dataset, our framework demonstrates remarkable privacy protection capabilities whilst maintaining practical deployability in real-world healthcare settings.

## Research Overview

### The Healthcare Privacy Challenge

Modern healthcare systems generate vast amounts of sensitive data through wearable devices, sensors, and monitoring equipment. Traditional machine learning approaches typically require centralised data repositories, which introduce substantial privacy risks and regulatory compliance challenges. Whilst federated learning offers a promising alternative by enabling model training across distributed devices without centralised data sharing, current implementations remain vulnerable to sophisticated privacy attacks that could expose confidential patient information.

Our research identified critical vulnerabilities in existing federated learning systems for healthcare applications, including membership inference attacks that can determine whether specific patient data participated in model training, gradient leakage attacks that can reconstruct private training data from model updates, and authentication vulnerabilities that enable unauthorised access to training processes.

### The FADES Solution

FADES addresses these challenges through a comprehensive, multi-layered security architecture that provides defence-in-depth protection whilst respecting the computational limitations of IoMT devices. Our framework follows a sequential workflow that systematically applies security mechanisms to protect patient privacy at every stage of the federated learning process.

The framework's architecture consists of four integrated components working in concert: Adaptive Differential Privacy dynamically adjusts noise injection based on data sensitivity metrics, effectively balancing privacy protection with model accuracy. Lightweight Homomorphic Encryption enables secure computation on encrypted model updates without requiring decryption, optimised specifically for the computational constraints of IoMT devices. Zero-Knowledge Proofs function as a verification layer, cryptographically confirming device authenticity and data integrity without exposing sensitive information. Finally, Federated Authentication combines autonomous authentication verification with centralised security access control, providing robust device verification.

## Key Research Contributions

### Privacy Protection Achievements

Our experimental validation demonstrates substantial improvements in privacy protection compared to baseline federated learning implementations. The framework successfully reduced membership inference attack effectiveness by an impressive 66%, decreasing the Area Under the Curve (AUC) from 0.61 to 0.21. This significant reduction means that adversaries attempting to determine whether specific patient data participated in model training face substantially reduced success rates, providing crucial protection for sensitive medical conditions.

Additionally, FADES degraded gradient leakage reconstruction quality by 53%, making it significantly more difficult for attackers to reconstruct private training data from model updates. Despite these robust security measures, our framework maintains 96% of the original model's accuracy, demonstrating that comprehensive privacy protection need not come at the expense of model utility.

### Computational Efficiency Optimisation

One of the most significant achievements of our research is demonstrating that comprehensive security measures can be implemented with minimal computational overhead. Our framework requires only 2.4 times the computational resources of baseline implementations, challenging common assumptions about the prohibitive costs of privacy-preserving techniques in resource-constrained environments.

This efficiency is achieved through several innovative optimisations: selective encryption targets only the most sensitive model parameters based on gradient sensitivity analysis, reducing encryption overhead by approximately 85% whilst maintaining security benefits. Adaptive noise injection adjusts protection levels based on real-time data characteristics rather than applying uniform privacy budgets, optimising the privacy-utility trade-off. Lightweight cryptographic implementations are specifically designed for IoMT device capabilities, ensuring practical deployability in real-world healthcare settings.

### Regulatory Compliance Framework

FADES is designed with regulatory compliance at its core, addressing requirements from both HIPAA (Health Insurance Portability and Accountability Act) in the United States and GDPR (General Data Protection Regulation) in Europe. The framework's multi-layered approach ensures that healthcare organisations can implement privacy-preserving collaborative learning whilst meeting stringent regulatory requirements for patient data protection.

## Dataset and Experimental Validation

### WESAD Dataset Overview

Our research utilises the Wearable Stress and Affect Detection (WESAD) dataset, a comprehensive multimodal physiological dataset that captures real-world healthcare monitoring scenarios. WESAD contains recordings from 15 participants across five different affective states: baseline, stress, amusement, meditation, and transient conditions. This dataset provides an ideal testing environment for our security framework as it represents the type of sensitive physiological data commonly collected by IoMT devices in clinical settings.

The dataset captures data from wrist-mounted Empatica E4 wearable devices, extracting four key physiological modalities: electrodermal activity (EDA) sampled at 4Hz, blood volume pulse (BVP) at 64Hz, skin temperature at 4Hz, and triaxial acceleration at 32Hz. This multimodal approach reflects the complexity of real-world IoMT data collection, where multiple sensors work together to provide comprehensive patient monitoring.

### Dataset Considerations and Accessibility

**Important Note for Researchers**: Due to the substantial size of the WESAD dataset (several gigabytes of physiological recordings), we have not included the raw data files in this repository. The complete WESAD dataset can be obtained from the original authors through their official channels. Our preprocessing pipeline and data handling scripts are designed to work seamlessly with the standard WESAD dataset format once downloaded and placed in the appropriate directory structure.

For researchers wishing to replicate our experiments, please download the WESAD dataset from the official source and place it in a `dataset/WESAD/` directory within the project root. Our preprocessing scripts will automatically handle data extraction, signal processing, and feature engineering as described in our methodology.

### Experimental Design and Subject Selection

From the 15 subjects available in the WESAD dataset, we strategically selected subjects S2 and S10 as representative samples for detailed analysis. This selection was based on three critical criteria: data completeness, with both subjects maintaining minimal sensor dropout (>98% data availability); signal quality characteristics, with distinct Signal-to-Noise Ratio properties (S2: 8.2dB, S10: 13.6dB) enabling evaluation of security mechanisms under varying physiological recording conditions; and activity distribution, providing sufficient samples across relevant affective states for comprehensive analysis.

This methodical subject selection enables thorough evaluation of our security framework's performance across different data characteristics whilst maintaining computational feasibility for extensive security testing. The varying signal quality between subjects also allows us to demonstrate the robustness of our adaptive privacy mechanisms across different data conditions.

## Technical Implementation

### Architecture Overview

FADES implements a client-server architecture specifically designed for privacy-preserving training across distributed wearable devices. The framework follows established federated learning protocols whilst incorporating our novel security enhancements at every stage of the training process. Our CNN architecture features carefully optimised layers designed for physiological signal classification, including convolutional layers with batch normalisation, global average pooling, and dense layers with dropout for regularisation.

The training configuration employs the Adam optimiser with a learning rate of 0.001, conducting 10 local epochs with early stopping mechanisms and 50 federated rounds with 40% client participation. This configuration balances training efficiency with convergence stability whilst accommodating the communication constraints typical in IoMT environments.

### Security Mechanisms Integration

Our adaptive differential privacy implementation uses layer-wise relevance propagation to determine optimal noise distribution across model parameters. The correlation ratio βj = |Rj|/∑|Rj| determines noise distribution based on feature relevance, with privacy budgets calculated as εj = βj × ε. This approach adds more noise to less relevant features whilst preserving the utility of critical model components.

The Paillier-based homomorphic encryption scheme selectively encrypts significant tensor values based on magnitude thresholds, enabling secure model aggregation through the homomorphic property E(m1)·E(m2) = E(m1+m2) mod n². This selective approach reduces computational overhead whilst maintaining security guarantees for the most sensitive model parameters.

Our zero-knowledge proof implementation uses HMAC-based verification to confirm client legitimacy without credential exposure. Authentication signatures are generated using Ω = HMAC(client_secret_bytes, challenge + client_id.encode('utf-8'), SHA256), with temporal verification ensuring authentication freshness through timestamp validation.

### Attack Resistance Validation

To demonstrate the effectiveness of our security measures, we implemented three sophisticated privacy attacks representative of real-world threats. Membership inference attacks attempt to determine whether specific data points participated in model training by analysing model confidence differences between training and non-training samples. Our implementation achieved 63% accuracy on baseline models but was reduced to only 50.5% success rate against fully secured implementations.

Gradient leakage attacks reconstruct training data samples from model gradients through iterative optimisation that minimises the distance between reconstructed and original gradients. Whilst baseline models showed concerning reconstruction quality with correlation coefficients of 0.031, our security measures significantly degraded attack effectiveness, increasing Mean Squared Error from 1.58 to 2.44.

Model inversion attacks extract class representations through optimisation techniques that maximise target class probability. Our framework reduced inversion success from 0.0053 to 0.3375 confidence scores, demonstrating substantial protection against attempts to reverse-engineer training data characteristics.

## Performance Analysis and Results

### Security-Performance Trade-offs

Our comprehensive evaluation reveals that FADES achieves an optimal balance between security and performance that makes it practical for deployment in real-world IoMT environments. The framework's modest computational overhead of only 1.4% training time increase challenges common assumptions about the prohibitive costs of implementing comprehensive security measures in resource-constrained settings.

Network communication analysis shows that our zero-knowledge proof verification adds only 4.2 ± 1.1 KB per client per round, whilst homomorphic encryption increases payload sizes by 312 ± 37%. For typical IoMT devices with limited bandwidth (approximately 250 Kbps for Bluetooth Low Energy-based medical sensors), these communication requirements remain within practical deployment parameters.

Cross-validation testing confirms that whilst differential privacy mechanisms can impact model utility (-22.92 ± 7.33 percentage points), homomorphic encryption maintains utility within statistical variance of baseline performance, and zero-knowledge proofs actually improve model performance by 25.00 ± 6.45 percentage points through improved client selection and quality assurance.

### Adaptive Privacy Mechanisms

One of the most significant findings from our research is the importance of adaptive privacy mechanisms that adjust protection levels based on data characteristics. Our analysis revealed that differential privacy effectiveness varies substantially based on Signal-to-Noise Ratio characteristics of the underlying physiological data. For subject S2 with lower SNR (8.2dB), differential privacy significantly decreased membership inference attack success, whilst for subject S10 with higher SNR (13.6dB), uniform noise application was less effective.

This finding highlights the limitations of one-size-fits-all privacy approaches and demonstrates the importance of our adaptive mechanisms that calibrate protection based on real-time data analysis. Future implementations should incorporate continuous sensitivity assessment rather than predetermined privacy budgets to optimise the privacy-utility trade-off across diverse patient populations and monitoring scenarios.

## Installation and Setup

### Prerequisites and Dependencies

FADES requires Python 3.8 or later along with several key dependencies for machine learning, cryptography, and data processing. The core machine learning framework relies on PyTorch 1.9+ for neural network implementation and federated learning orchestration. Data processing utilises NumPy for numerical computations and Scikit-learn for preprocessing and evaluation metrics. Cryptographic operations require specialised libraries for homomorphic encryption and zero-knowledge proof implementations.

### Environment Configuration

```bash
# Clone the repository
git clone https://github.com/ahmedulster/FADES.git
cd FADES

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download WESAD dataset (follow official instructions)
# Place dataset in dataset/WESAD/ directory
```

### Dataset Preparation

Due to licensing and size constraints, researchers must independently obtain the WESAD dataset from the original authors. Once downloaded, place the dataset files in the `dataset/WESAD/` directory following the standard WESAD structure. Our preprocessing pipeline includes automatic data validation and will alert users if the dataset structure does not match expected formats.

The preprocessing pipeline handles signal extraction, windowing with 50% overlap, SMOTE oversampling for class balance correction, and min-max normalisation. Data is automatically partitioned into training (75%) and testing (25%) sets using stratified sampling to ensure representative distribution across affective states and subjects.

## Usage Examples and Applications

### Basic Federated Training

To run federated learning with default settings, simply execute the main training script. The system will automatically configure client-server communication, initialise the CNN model, and begin federated training across simulated IoMT devices.

```bash
# Basic federated learning without security
python federated_training.py

# With comprehensive security measures
python federated_training.py --use_dp --use_he --use_zkp --use_auth

# Custom privacy parameters
python federated_training.py --epsilon 3.0 --delta 1e-5 --he_key_size 2048
```

### Security Evaluation and Testing

Our framework includes comprehensive security evaluation tools that enable researchers to assess privacy protection effectiveness and validate security measures against various attack vectors.

```bash
# Run individual attack simulations
python attacks/membership_inference.py --target_model saved_models/baseline_model.pt
python attacks/gradient_leakage.py --reconstruction_steps 500
python attacks/model_inversion.py --target_class stress

# Comprehensive security evaluation
python benchmarks/security_analysis.py --full_evaluation
python benchmarks/privacy_metrics.py --generate_report
```

### Performance Benchmarking

The benchmarking suite provides detailed analysis of computational overhead, communication costs, and privacy-utility trade-offs across different security configurations.

```bash
# Performance profiling
python benchmarks/performance_profiler.py --config all_security
python benchmarks/communication_analysis.py --simulate_network_conditions

# Comparative analysis across configurations
python benchmarks/comparative_analysis.py --compare_all_configs
```

## Research Applications and Extensions

### Healthcare Research Applications

FADES provides a robust foundation for various healthcare research applications that require privacy-preserving collaborative learning. Stress detection and mental health monitoring can benefit from our framework's ability to train accurate models across distributed wearable devices whilst protecting sensitive psychological and physiological data. Chronic disease management systems can leverage FADES to develop personalised treatment recommendations without compromising patient privacy.

Clinical trial data analysis represents another significant application area, where our framework enables multi-institutional collaboration whilst maintaining regulatory compliance and protecting participant confidentiality. Remote patient monitoring systems can implement FADES to improve diagnostic accuracy through collaborative learning whilst ensuring HIPAA and GDPR compliance.

### Research Extensions and Future Directions

Our research opens several promising directions for future investigation. Adaptive security mechanisms that dynamically adjust protection levels based on real-time threat assessment and data sensitivity analysis could further optimise the privacy-utility trade-off. Sparse update mechanisms could reduce communication overhead whilst maintaining security guarantees, making the framework more suitable for bandwidth-constrained IoMT environments.

Advanced differential privacy implementations with adaptive privacy budgets based on patient population characteristics could provide more nuanced protection that accounts for varying sensitivity levels across different demographic groups and medical conditions. Integration with edge computing architectures could enable local model aggregation whilst maintaining our security guarantees, reducing dependency on centralised servers.

### Regulatory Compliance and Deployment

FADES is designed to support healthcare organisations in meeting stringent regulatory requirements whilst enabling innovative collaborative learning applications. The framework's documentation includes guidance for HIPAA compliance assessments, GDPR impact evaluations, and clinical deployment considerations.

For organisations considering deployment, we recommend conducting thorough security audits, establishing clear data governance policies, and implementing comprehensive monitoring systems to ensure ongoing compliance and security effectiveness. Our framework provides the technical foundation, but successful deployment requires careful attention to organisational policies and regulatory requirements.

## Citation and Academic Use

If you use FADES in your research, please cite our work:

```bibtex
@misc{ahmed2024fades,
  title={FADES: Federated Authentication with Adaptive Differential Privacy, Encryption, and Zero-Knowledge Proofs for Secure IoMT Devices},
  author={Ahmed, Ayesha},
  institution={School of Computing, Ulster University},
  year={2024},
  howpublished={\url{https://github.com/ahmedulster/FADES}}
}
```

## Contributing and Collaboration

We welcome contributions from the research community to enhance FADES and extend its capabilities. Potential contribution areas include optimisation of cryptographic implementations for resource-constrained devices, development of additional privacy-preserving mechanisms, extension to new healthcare datasets and applications, and improvement of computational efficiency.

Contributors should follow our established coding standards, include comprehensive tests for new functionality, update documentation for any changes, and ensure compatibility with existing security mechanisms. All contributions undergo thorough security review to maintain the framework's integrity and effectiveness.

## Acknowledgements and Support

This research was conducted at the School of Computing, Ulster University, with support from healthcare technology partners and the privacy-preserving machine learning research community. We acknowledge the creators of the WESAD dataset for providing a comprehensive evaluation platform and the federated learning research community for foundational frameworks that inspired our work.

For technical support, research collaboration opportunities, or deployment guidance, please use our GitHub Issues system or contact the research team directly. We are committed to supporting the healthcare technology community in implementing privacy-preserving collaborative learning solutions.

## Licence and Legal Considerations

FADES is released under the MIT License, providing flexibility for both academic research and commercial applications. However, users should note that deployment in clinical settings requires additional considerations including regulatory approval, security auditing, and compliance verification. This research framework provides a technical foundation but does not constitute medical advice or replace professional healthcare expertise.

Healthcare organisations considering deployment should conduct independent security assessments, ensure compliance with applicable regulations, and implement appropriate safeguards for patient data protection. Whilst our research demonstrates significant privacy improvements, no security system is completely impervious to all possible attacks, and ongoing monitoring and updates are essential for maintaining protection effectiveness.

---

**Research Contact**: Ahmed B00947072, School of Computing, Ulster University, Belfast, UK
**Repository**: https://github.com/ahmedulster/FADES
**Documentation**: Comprehensive technical documentation and experimental details available in the repository
