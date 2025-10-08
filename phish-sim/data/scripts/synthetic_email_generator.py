# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Phish-Sim Project

"""
Synthetic email content generator for phishing and benign emails
"""

import random
import string
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import SYNTHETIC_CONFIG, get_data_path

logger = logging.getLogger(__name__)

class SyntheticEmailGenerator:
    """Generate synthetic phishing and benign emails"""
    
    def __init__(self):
        self.phishing_templates = SYNTHETIC_CONFIG["emails"]["templates"]["phishing"]
        self.benign_templates = SYNTHETIC_CONFIG["emails"]["templates"]["benign"]
        
        # Common phishing email templates
        self.phishing_content = {
            "urgent_security_alert": {
                "subject": "URGENT: Security Alert - Account Compromise Detected",
                "body": """Dear Valued Customer,

We have detected suspicious activity on your account. For your security, we need to verify your identity immediately.

Click here to verify your account: {malicious_link}

If you do not verify within 24 hours, your account will be suspended.

This is an automated message. Please do not reply to this email.

Best regards,
Security Team
{company_name}""",
                "urgency_indicators": ["URGENT", "immediately", "24 hours", "suspended"]
            },
            
            "account_verification": {
                "subject": "Account Verification Required - Action Needed",
                "body": """Hello {user_name},

Your account requires immediate verification to continue using our services.

Please click the link below to verify your account:
{malicious_link}

This verification is mandatory and must be completed within 48 hours.

If you have any questions, please contact our support team.

Thank you,
Account Services
{company_name}""",
                "urgency_indicators": ["immediate", "mandatory", "48 hours"]
            },
            
            "payment_confirmation": {
                "subject": "Payment Confirmation - Please Review",
                "body": """Dear Customer,

We have processed a payment of ${amount} from your account.

Transaction Details:
- Amount: ${amount}
- Date: {date}
- Reference: {reference}

If you did not authorize this payment, please click here to dispute:
{malicious_link}

This is an automated notification. Please do not reply.

Customer Service
{company_name}""",
                "urgency_indicators": ["dispute", "unauthorized"]
            },
            
            "password_reset": {
                "subject": "Password Reset Request - Security Notice",
                "body": """Hello,

We received a request to reset your password. If you made this request, click the link below:

{malicious_link}

If you did not request a password reset, please ignore this email or contact support immediately.

For security reasons, this link will expire in 2 hours.

Security Team
{company_name}""",
                "urgency_indicators": ["immediately", "expire", "2 hours"]
            }
        }
        
        # Benign email templates
        self.benign_content = {
            "newsletter": {
                "subject": "Weekly Newsletter - Latest Updates",
                "body": """Hi {user_name},

Here are this week's highlights:

• New features released
• Community updates
• Upcoming events
• Tips and tutorials

Read more on our blog: {legitimate_link}

Thank you for being part of our community!

Best regards,
The Team
{company_name}""",
                "urgency_indicators": []
            },
            
            "order_confirmation": {
                "subject": "Order Confirmation - #{order_number}",
                "body": """Dear {user_name},

Thank you for your order! Here are the details:

Order Number: #{order_number}
Date: {date}
Total: ${amount}

Items:
{items}

Your order will be processed within 1-2 business days.

Track your order: {legitimate_link}

Customer Service
{company_name}""",
                "urgency_indicators": []
            },
            
            "meeting_invitation": {
                "subject": "Meeting Invitation - {meeting_topic}",
                "body": """Hello {user_name},

You are invited to a meeting:

Topic: {meeting_topic}
Date: {date}
Time: {time}
Location: {location}

Please confirm your attendance by replying to this email.

Best regards,
{organizer_name}""",
                "urgency_indicators": []
            },
            
            "system_notification": {
                "subject": "System Maintenance Notice",
                "body": """Dear Users,

We will be performing scheduled system maintenance:

Date: {date}
Time: {time}
Duration: {duration}

During this time, some services may be temporarily unavailable.

We apologize for any inconvenience.

IT Department
{company_name}""",
                "urgency_indicators": []
            }
        }
        
        # Company names and domains
        self.companies = [
            "Microsoft", "Google", "Apple", "Amazon", "Facebook", "Twitter",
            "LinkedIn", "PayPal", "eBay", "Netflix", "Spotify", "Dropbox"
        ]
        
        self.domains = [
            "microsoft.com", "google.com", "apple.com", "amazon.com",
            "facebook.com", "twitter.com", "linkedin.com", "paypal.com"
        ]
    
    def generate_phishing_email(self, template_type: str = None) -> Dict[str, Any]:
        """Generate a phishing email"""
        if template_type is None:
            template_type = random.choice(self.phishing_templates)
        
        template = self.phishing_content[template_type]
        company = random.choice(self.companies)
        
        # Generate malicious link
        malicious_links = [
            f"http://{company.lower()}-security-verify.net/login",
            f"https://secure-{company.lower()}-account.com/verify",
            f"http://{company.lower()}-account-update.org/login",
            f"https://{company.lower()}-security-alert.com/verify"
        ]
        malicious_link = random.choice(malicious_links)
        
        # Generate fake user data
        user_name = f"User{random.randint(1000, 9999)}"
        amount = f"{random.randint(50, 5000)}.{random.randint(10, 99)}"
        date = (datetime.now() - timedelta(days=random.randint(0, 7))).strftime("%Y-%m-%d")
        reference = f"TXN{random.randint(100000, 999999)}"
        
        # Format email content
        subject = template["subject"]
        body = template["body"].format(
            user_name=user_name,
            malicious_link=malicious_link,
            company_name=company,
            amount=amount,
            date=date,
            reference=reference
        )
        
        # Generate sender email
        sender_domains = [
            f"noreply@{company.lower()}-security.com",
            f"security@{company.lower()}-verify.net",
            f"alerts@{company.lower()}-account.org",
            f"support@{company.lower()}-security.net"
        ]
        sender = random.choice(sender_domains)
        
        return {
            'subject': subject,
            'body': body,
            'sender': sender,
            'recipient': f"user{random.randint(1000, 9999)}@example.com",
            'label': 'phish',
            'confidence': random.uniform(0.8, 1.0),
            'template_type': template_type,
            'company': company,
            'source': 'synthetic_phishing',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'subject_length': len(subject),
                'body_length': len(body),
                'has_urgency_words': any(word in subject.upper() + body.upper() for word in template["urgency_indicators"]),
                'has_malicious_link': True,
                'sender_domain_legitimacy': 0.2,  # Low legitimacy score
                'spelling_errors': random.randint(0, 3)
            }
        }
    
    def generate_benign_email(self, template_type: str = None) -> Dict[str, Any]:
        """Generate a benign email"""
        if template_type is None:
            template_type = random.choice(self.benign_templates)
        
        template = self.benign_content[template_type]
        company = random.choice(self.companies)
        domain = random.choice(self.domains)
        
        # Generate legitimate link
        legitimate_links = [
            f"https://{domain}/support",
            f"https://{domain}/help",
            f"https://{domain}/blog",
            f"https://{domain}/account"
        ]
        legitimate_link = random.choice(legitimate_links)
        
        # Generate user data
        user_name = f"User{random.randint(1000, 9999)}"
        order_number = f"ORD{random.randint(100000, 999999)}"
        amount = f"{random.randint(20, 500)}.{random.randint(10, 99)}"
        date = (datetime.now() - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        time = f"{random.randint(9, 17):02d}:{random.randint(0, 59):02d}"
        duration = f"{random.randint(1, 4)} hours"
        meeting_topic = random.choice([
            "Project Review", "Team Meeting", "Client Discussion", "Sprint Planning"
        ])
        location = random.choice([
            "Conference Room A", "Zoom Meeting", "Office 101", "Virtual Meeting"
        ])
        organizer_name = f"Manager{random.randint(1, 10)}"
        items = "\n".join([
            f"- Item {i+1}: ${random.randint(10, 100)}.{random.randint(10, 99)}"
            for i in range(random.randint(1, 5))
        ])
        
        # Format email content
        subject = template["subject"].format(
            order_number=order_number,
            meeting_topic=meeting_topic
        )
        body = template["body"].format(
            user_name=user_name,
            legitimate_link=legitimate_link,
            company_name=company,
            order_number=order_number,
            amount=amount,
            date=date,
            time=time,
            duration=duration,
            meeting_topic=meeting_topic,
            location=location,
            organizer_name=organizer_name,
            items=items
        )
        
        # Generate sender email
        sender = f"noreply@{domain}"
        
        return {
            'subject': subject,
            'body': body,
            'sender': sender,
            'recipient': f"user{random.randint(1000, 9999)}@example.com",
            'label': 'benign',
            'confidence': random.uniform(0.9, 1.0),
            'template_type': template_type,
            'company': company,
            'source': 'synthetic_benign',
            'timestamp': datetime.now().isoformat(),
            'features': {
                'subject_length': len(subject),
                'body_length': len(body),
                'has_urgency_words': False,
                'has_malicious_link': False,
                'sender_domain_legitimacy': 0.9,  # High legitimacy score
                'spelling_errors': 0
            }
        }
    
    def generate_dataset(self, count: int, phishing_ratio: float = 0.4) -> List[Dict[str, Any]]:
        """Generate a dataset of synthetic emails"""
        logger.info(f"Generating {count} synthetic emails with {phishing_ratio*100}% phishing ratio")
        
        phishing_count = int(count * phishing_ratio)
        benign_count = count - phishing_count
        
        dataset = []
        
        # Generate phishing emails
        for _ in range(phishing_count):
            template_type = random.choice(self.phishing_templates)
            email_data = self.generate_phishing_email(template_type)
            dataset.append(email_data)
        
        # Generate benign emails
        for _ in range(benign_count):
            template_type = random.choice(self.benign_templates)
            email_data = self.generate_benign_email(template_type)
            dataset.append(email_data)
        
        # Shuffle the dataset
        random.shuffle(dataset)
        
        logger.info(f"Generated {len(dataset)} emails: {phishing_count} phishing, {benign_count} benign")
        return dataset

def main():
    """Main function to generate synthetic email dataset"""
    generator = SyntheticEmailGenerator()
    
    # Generate dataset
    config = SYNTHETIC_CONFIG["emails"]
    dataset = generator.generate_dataset(
        count=config["count"],
        phishing_ratio=config["phishing_ratio"]
    )
    
    # Save dataset
    import pandas as pd
    df = pd.DataFrame(dataset)
    
    output_path = get_data_path('synthetic', 'synthetic_emails.csv')
    df.to_csv(output_path, index=False)
    
    logger.info(f"Synthetic email dataset saved to {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("SYNTHETIC EMAIL GENERATION SUMMARY")
    print("="*50)
    print(f"Total Emails: {len(dataset)}")
    print(f"Phishing Emails: {len([d for d in dataset if d['label'] == 'phish'])}")
    print(f"Benign Emails: {len([d for d in dataset if d['label'] == 'benign'])}")
    
    # Template distribution
    templates = {}
    for item in dataset:
        template = item['template_type']
        templates[template] = templates.get(template, 0) + 1
    
    print("\nTemplate Types Used:")
    for template, count in templates.items():
        print(f"  {template}: {count}")
    print("="*50)

if __name__ == "__main__":
    main()