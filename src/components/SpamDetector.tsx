import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { SAMPLE_EMAILS } from "@/lib/spamDetector";
import { Shield, AlertTriangle, CheckCircle, Mail, Sparkles, Activity } from "lucide-react";
import { toast } from "sonner";

interface MLAnalysis {
  isSpam: boolean;
  confidence: number;
  spamProbability: number;
  prediction: string;
}

// API endpoint - change this if deploying Flask API elsewhere
const API_URL = "http://localhost:5000";

export default function SpamDetector() {
  const [emailText, setEmailText] = useState("");
  const [analysis, setAnalysis] = useState<MLAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiStatus, setApiStatus] = useState<'unknown' | 'connected' | 'error'>('unknown');

  const handleAnalyze = async () => {
    if (!emailText.trim()) {
      toast.error("Please enter email text to analyze");
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: emailText }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const result = await response.json();
      setAnalysis(result);
      setApiStatus('connected');
      
      if (result.isSpam) {
        toast.error("Spam detected!", {
          description: `Confidence: ${(result.confidence * 100).toFixed(1)}%`
        });
      } else {
        toast.success("Email appears safe", {
          description: `Confidence: ${(result.confidence * 100).toFixed(1)}%`
        });
      }
    } catch (error) {
      console.error('Error calling Flask API:', error);
      setApiStatus('error');
      toast.error("API Connection Error", {
        description: "Make sure Flask API is running on http://localhost:5000"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const loadSampleEmail = (type: 'spam' | 'legitimate', index: number) => {
    const samples = SAMPLE_EMAILS[type];
    const sample = samples[index];
    const fullEmail = `Subject: ${sample.subject}\n\n${sample.body}`;
    setEmailText(fullEmail);
    setAnalysis(null);
    toast.info("Sample email loaded");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted">
      <div className="container mx-auto px-4 py-12 max-w-6xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-primary to-accent blur-2xl opacity-30 rounded-full"></div>
              <div className="relative bg-card p-4 rounded-2xl shadow-lg border border-border">
                <Shield className="w-12 h-12 text-primary" />
              </div>
            </div>
          </div>
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
            Email Spam Detector
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            ML-powered spam detection using trained Logistic Regression model
          </p>
          <div className="flex items-center justify-center gap-3 mt-4 flex-wrap">
            <Badge variant={apiStatus === 'connected' ? 'default' : apiStatus === 'error' ? 'destructive' : 'secondary'} className="px-3 py-1">
              <Activity className="w-3 h-3 mr-1" />
              {apiStatus === 'connected' ? 'API Connected' : apiStatus === 'error' ? 'API Disconnected' : 'API Status Unknown'}
            </Badge>
            <Badge variant="outline" className="px-3 py-1">F1 Score: 96.8%</Badge>
            <Badge variant="outline" className="px-3 py-1">Accuracy: 98.2%</Badge>
            <Badge variant="outline" className="px-3 py-1">3000 Features</Badge>
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8 mb-12">
          {/* Main Analysis Panel */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="shadow-xl border-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Mail className="w-5 h-5" />
                  Enter Email Content
                </CardTitle>
                <CardDescription>
                  Paste the email content below to analyze for spam indicators
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Textarea
                  placeholder="Subject: Amazing Offer!&#10;&#10;Dear Friend,&#10;&#10;You have won $1,000,000! Click here to claim..."
                  value={emailText}
                  onChange={(e) => setEmailText(e.target.value)}
                  className="min-h-[300px] font-mono text-sm resize-none"
                />
                <div className="flex gap-3">
                  <Button 
                    onClick={handleAnalyze}
                    disabled={isAnalyzing || !emailText.trim()}
                    className="flex-1 h-12 text-lg shadow-lg"
                    size="lg"
                  >
                    {isAnalyzing ? (
                      <>
                        <Sparkles className="w-5 h-5 mr-2 animate-spin" />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <Shield className="w-5 h-5 mr-2" />
                        Analyze Email
                      </>
                    )}
                  </Button>
                  <Button 
                    variant="outline" 
                    onClick={() => { setEmailText(""); setAnalysis(null); }}
                    disabled={!emailText}
                    size="lg"
                    className="h-12"
                  >
                    Clear
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Results Panel */}
            {analysis && (
              <Card className={`shadow-xl border-2 ${
                analysis.isSpam 
                  ? 'border-destructive/50 bg-destructive/5' 
                  : 'border-success/50 bg-success/5'
              }`}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2 text-2xl">
                        {analysis.isSpam ? (
                          <>
                            <AlertTriangle className="w-6 h-6 text-destructive" />
                            <span className="text-destructive">Spam Detected</span>
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-6 h-6 text-success" />
                            <span className="text-success">Email Appears Safe</span>
                          </>
                        )}
                      </CardTitle>
                      <CardDescription className="mt-2">
                        Analysis completed with {(analysis.confidence * 100).toFixed(1)}% confidence
                      </CardDescription>
                    </div>
                    <Badge 
                      variant={analysis.isSpam ? "destructive" : "default"}
                      className="text-lg px-4 py-2"
                    >
                      {(analysis.confidence * 100).toFixed(1)}%
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Confidence Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">Overall Confidence</div>
                      <div className="text-3xl font-bold">{(analysis.confidence * 100).toFixed(1)}%</div>
                      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                        <div 
                          className="h-full rounded-full bg-primary transition-all duration-1000"
                          style={{ width: `${analysis.confidence * 100}%` }}
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">Spam Probability</div>
                      <div className={`text-3xl font-bold ${analysis.isSpam ? 'text-destructive' : 'text-success'}`}>
                        {(analysis.spamProbability * 100).toFixed(1)}%
                      </div>
                      <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
                        <div 
                          className={`h-full rounded-full transition-all duration-1000 ${
                            analysis.isSpam 
                              ? 'bg-gradient-to-r from-destructive to-destructive/80' 
                              : 'bg-gradient-to-r from-success to-success/80'
                          }`}
                          style={{ width: `${analysis.spamProbability * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Model Info */}
                  <Alert>
                    <Sparkles className="w-4 h-4" />
                    <AlertDescription>
                      Prediction made by trained <strong>Logistic Regression</strong> model with 98.2% accuracy and 96.8% F1 score.
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Sample Emails */}
            <Card className="shadow-lg">
              <CardHeader>
                <CardTitle className="text-lg">Try Sample Emails</CardTitle>
                <CardDescription>
                  Test the detector with pre-loaded examples
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-destructive flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    Spam Examples
                  </h4>
                  {SAMPLE_EMAILS.spam.map((email, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      className="w-full justify-start text-left h-auto py-3"
                      onClick={() => loadSampleEmail('spam', index)}
                    >
                      <div className="text-xs">
                        <div className="font-semibold truncate">{email.subject}</div>
                        <div className="text-muted-foreground truncate">{email.body.substring(0, 40)}...</div>
                      </div>
                    </Button>
                  ))}
                </div>

                <div className="space-y-2">
                  <h4 className="text-sm font-semibold text-success flex items-center gap-2">
                    <CheckCircle className="w-4 h-4" />
                    Legitimate Examples
                  </h4>
                  {SAMPLE_EMAILS.legitimate.map((email, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      className="w-full justify-start text-left h-auto py-3"
                      onClick={() => loadSampleEmail('legitimate', index)}
                    >
                      <div className="text-xs">
                        <div className="font-semibold truncate">{email.subject}</div>
                        <div className="text-muted-foreground truncate">{email.body.substring(0, 40)}...</div>
                      </div>
                    </Button>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Model Info Card */}
            <Card className="shadow-lg bg-primary/5 border-primary/20">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-primary" />
                  Model Information
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Algorithm:</span>
                    <span className="font-semibold">Logistic Regression</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">F1 Score:</span>
                    <span className="font-semibold text-success">96.8%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Accuracy:</span>
                    <span className="font-semibold text-success">98.2%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Precision:</span>
                    <span className="font-semibold">96.4%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Recall:</span>
                    <span className="font-semibold">97.3%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Features:</span>
                    <span className="font-semibold">3000 TF-IDF</span>
                  </div>
                </div>
                <Alert className="mt-4">
                  <Activity className="w-4 h-4" />
                  <AlertDescription className="text-xs">
                    Make sure Flask API is running on <code className="bg-muted px-1 py-0.5 rounded">localhost:5000</code>
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
