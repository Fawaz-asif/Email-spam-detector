// Simple client-side spam detection prototype
// For production, integrate with ML API via Lovable Cloud

const SPAM_KEYWORDS = [
  'winner', 'congratulations', 'claim', 'prize', 'free money', 'cash bonus',
  'click here', 'act now', 'limited time', 'urgent', 'lottery', 'million dollars',
  'nigerian prince', 'inheritance', 'bank account', 'verify account', 'suspended',
  'bitcoin', 'cryptocurrency investment', 'double your money', 'guaranteed income',
  'work from home', 'make money fast', 'no experience needed', 'weight loss',
  'pharmacy', 'viagra', 'cialis', 'drugs', 'prescription', 'unsubscribe',
  'dear friend', 'business proposal', 'confidential', 'transfer funds'
];

const SPAM_PATTERNS = [
  /\$\d+[,\d]*(\.\d{2})?/g, // Money amounts
  /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, // Phone numbers
  /\b[A-Z]{5,}\b/g, // ALL CAPS words
  /!{2,}/g, // Multiple exclamation marks
  /\?{2,}/g, // Multiple question marks
  /https?:\/\/bit\.ly/gi, // Shortened URLs
  /click.*here/gi, // "Click here" variations
];

export interface SpamAnalysis {
  isSpam: boolean;
  confidence: number;
  indicators: string[];
  score: number;
}

export function analyzeEmail(text: string): SpamAnalysis {
  if (!text || text.trim().length === 0) {
    return {
      isSpam: false,
      confidence: 0,
      indicators: [],
      score: 0
    };
  }

  const lowerText = text.toLowerCase();
  const indicators: string[] = [];
  let score = 0;

  // Check for spam keywords
  SPAM_KEYWORDS.forEach(keyword => {
    if (lowerText.includes(keyword)) {
      indicators.push(`Contains spam keyword: "${keyword}"`);
      score += 10;
    }
  });

  // Check for spam patterns
  SPAM_PATTERNS.forEach(pattern => {
    const matches = text.match(pattern);
    if (matches && matches.length > 0) {
      if (pattern.source.includes('A-Z')) {
        indicators.push(`Excessive capitalization detected`);
        score += 5;
      } else if (pattern.source.includes('!') || pattern.source.includes('?')) {
        indicators.push(`Excessive punctuation detected`);
        score += 5;
      } else if (pattern.source.includes('bit.ly')) {
        indicators.push(`Shortened URL detected`);
        score += 15;
      } else if (pattern.source.includes('click')) {
        indicators.push(`Suspicious call-to-action detected`);
        score += 10;
      } else {
        score += 5;
      }
    }
  });

  // Additional heuristics
  const hasMultipleLinks = (text.match(/https?:\/\//g) || []).length > 3;
  if (hasMultipleLinks) {
    indicators.push(`Multiple links detected`);
    score += 15;
  }

  const hasExcessiveCaps = (text.match(/[A-Z]/g) || []).length / text.length > 0.3;
  if (hasExcessiveCaps && text.length > 50) {
    indicators.push(`Excessive use of capital letters`);
    score += 10;
  }

  const wordCount = text.split(/\s+/).length;
  if (wordCount < 10) {
    score -= 10; // Very short messages less likely to be spam
  }

  // Calculate confidence (0-100)
  const normalizedScore = Math.min(score, 100);
  const confidence = Math.round(normalizedScore);
  const isSpam = confidence >= 50;

  // Add verdict indicator
  if (indicators.length === 0) {
    indicators.push(isSpam ? 'General spam characteristics detected' : 'No spam indicators found');
  }

  return {
    isSpam,
    confidence,
    indicators: indicators.slice(0, 5), // Limit to top 5 indicators
    score: normalizedScore
  };
}

export const SAMPLE_EMAILS = {
  spam: [
    {
      subject: "CONGRATULATIONS! You've Won $1,000,000!",
      body: "Dear Winner,\n\nCONGRATULATIONS!!! You have been selected as the LUCKY WINNER of our international lottery! You've won $1,000,000 USD!!!\n\nACT NOW to claim your prize! Click here immediately: http://bit.ly/fake-link\n\nThis is a LIMITED TIME OFFER! Respond within 24 hours or your prize will be given to someone else!\n\nProvide your bank account details to transfer the funds.\n\nBest regards,\nInternational Lottery Committee"
    },
    {
      subject: "Urgent: Your Account Has Been Suspended",
      body: "Dear User,\n\nYour account has been suspended due to suspicious activity. Click here to verify your account immediately: http://bit.ly/verify-now\n\nFailure to verify within 24 hours will result in permanent account closure.\n\nProvide your login credentials and bank information to restore access.\n\nSecurity Team"
    }
  ],
  legitimate: [
    {
      subject: "Meeting Reminder - Q4 Planning",
      body: "Hi Team,\n\nThis is a friendly reminder about our Q4 planning meeting scheduled for tomorrow at 2 PM in Conference Room B.\n\nAgenda:\n- Review Q3 results\n- Set Q4 objectives\n- Budget allocation\n\nPlease bring your department reports. Let me know if you have any questions.\n\nBest regards,\nSarah"
    },
    {
      subject: "Your Order Confirmation #12345",
      body: "Thank you for your order!\n\nOrder Details:\n- Product: Wireless Headphones\n- Quantity: 1\n- Total: $79.99\n\nYour order will ship within 2-3 business days. You can track your shipment at www.example.com/tracking\n\nIf you have any questions, please contact our support team.\n\nThank you for shopping with us!"
    }
  ]
};
