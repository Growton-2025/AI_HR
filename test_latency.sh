for i in {1..5}
do
  curl -w "Time: %{time_total}s\n" -s "http://127.0.0.1:8000/api/outreach/chat/linkedin/0/13096" | tail -n 1
done
