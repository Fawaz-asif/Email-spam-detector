import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { analyzeEmail, SAMPLE_EMAILS, type SpamAnalysis } from "@/lib/spamDetector";
import { Shield, AlertTriangle, CheckCircle, Mail, Sparkles } from "lucide-react";
import { toast } from "sonner";

export default function SpamDetector() {
  const [emailText, setEmailText] = useState("");
  const [analysis, setAnalysis] = useState<SpamAnalysis | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = () => {
    if (!emailText.trim()) {
      toast.error("Please enter email text to analyze");
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate API delay for better UX
    setTimeout(() => {
      const result = analyzeEmail(emailText);
      setAnalysis(result);
      setIsAnalyzing(false);
      
      if (result.isSpam) {
        toast.error("Spam detected!", {
          description: `Confidence: ${result.confidence}%`
        });
      } else {
        toast.success("Email appears safe", {
          description: `Confidence: ${100 - result.confidence}%`
        });
      }
    }, 800);
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
            Advanced AI-powered spam detection to keep your inbox safe and secure
          </p>
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
                        Analysis completed with {analysis.confidence}% confidence
                      </CardDescription>
                    </div>
                    <Badge 
                      variant={analysis.isSpam ? "destructive" : "default"}
                      className="text-lg px-4 py-2"
                    >
                      {analysis.confidence}%
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Confidence Bar */}
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Spam Probability</span>
                      <span className="font-semibold">{analysis.confidence}%</span>
                    </div>
                    <div className="w-full bg-muted rounded-full h-3 overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-1000 ${
                          analysis.isSpam 
                            ? 'bg-gradient-to-r from-destructive to-destructive/80' 
                            : 'bg-gradient-to-r from-success to-success/80'
                        }`}
                        style={{ width: `${analysis.confidence}%` }}
                      />
                    </div>
                  </div>

                  {/* Indicators */}
                  <div className="space-y-3">
                    <h4 className="font-semibold flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4" />
                      Detection Indicators
                    </h4>
                    <div className="space-y-2">
                      {analysis.indicators.map((indicator, index) => (
                        <Alert key={index} variant={analysis.isSpam ? "destructive" : "default"}>
                          <AlertDescription className="flex items-start gap-2">
                            <span className="text-lg">•</span>
                            <span>{indicator}</span>
                          </AlertDescription>
                        </Alert>
                      ))}
                    </div>
                  </div>
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

            {/* Info Card */}
            <Card className="shadow-lg bg-primary/5 border-primary/20">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-primary" />
                  About This Detector
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <p>
                  This is a prototype using client-side pattern matching and keyword analysis.
                </p>
                <p className="font-semibold text-foreground">
                  For production ML models:
                </p>
                <ul className="space-y-1 list-disc list-inside">
                  <li>Integrate with external ML APIs</li>
                  <li>Use Lovable Cloud for backend</li>
                  <li>Deploy trained models (TensorFlow.js)</li>
                  <li>Real-time learning capabilities</li>
                </ul>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
