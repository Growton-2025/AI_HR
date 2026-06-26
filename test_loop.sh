> /tmp/ai_hr_backend.log
for i in {1..3}
do
  echo "Curl $i:"
  curl -w "Time: %{time_total}s\n" -s "http://127.0.0.1:8000/api/outreach/chat/linkedin/0/13096?force=true" | tail -n 1
done
echo "Logs:"
grep "DEBUG TIMING" /tmp/ai_hr_backend.log
